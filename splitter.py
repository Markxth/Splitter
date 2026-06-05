import streamlit as st
from PIL import Image
from time import sleep
import cv2 as cv
import numpy as np 
from dotenv import find_dotenv, load_dotenv
import torch 
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

load_dotenv(find_dotenv()) 
device = "cuda" if torch.cuda.is_available() else "cpu" 
@st.cache_resource #so we load it once and that is it. 
def load_model() : 
    dtype = torch.bfloat16 if device == "cuda" else torch.float32 #so we do not break the code if the user has CPU only.
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            torch_dtype = dtype,  
            device_map="auto" #if one wants to use low_mem_cpu_usage = True, then remove this line and uncomment the line belw. Do not have both as it will create race conditions from time to time
            #low_cpu_mem_usage = True
        )

    processor = AutoProcessor.from_pretrained(model_id) #load processor
    return model, processor

def run_qwen(panel, processor, model, text_input) : 
   
    messages = [
        {
            "role" : "user",
            "content": [
                {
                    "type" : "image" ,
                    "image" : panel
                },
                {
                    "type" :  "text" ,
                    "text" : text_input
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True #tokenize is False so we can later pass the actual string into the processor, else it will throw an error
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor( #this function does not have videos as we do not care about handling videos.
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device) 

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=600)
        
        # Trim the prompt 
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

    return [(output_text[0])]

def vertical_split_sort(panel, image_area, image_y) : #this one is for the cases when the image is split badly vertically
    panel_area = panel.shape[0] * panel.shape[1] #height x width
    if (panel_area < 0.25 * image_area) : 
        return [panel] #make it a list

    rows_bright = np.mean(panel, axis= 1 ) 
    bright_average = np.where(rows_bright > 200 )[0] #the value after ">" can be changed depending on context
    
    width_bright = [] #adding a row size check as panels with lines within them mess up the panel splitter
    for row in bright_average : 
        row_tbd = panel[row, :] 
        pixels_bright = np.sum(row_tbd > 200)
        if pixels_bright > 0.95 * panel.shape[1] : 
            width_bright.append(row)

        bright_average = np.array(width_bright)  

    lists = [] 
    cluster_start =0  

    #find cluster of bright rows
    for x in range(1, len(bright_average)) : 
        if(bright_average[x] - bright_average[x-1] > 10 ) : 
            lists.append( (cluster_start + bright_average[x])  // 2 ) #for a simple thin line it is unnecessary but for more complex images it is beneficial to filter clusters of numbers
            cluster_start = bright_average[x] 
    
    if not lists : 
        return None 
    results = []
    prev = 0 
    for i in lists : 
        results.append(panel[prev:i , :])
        prev = i
    results.append(panel[prev:, :]) 

def split_sort(contours, image):
    if contours is None : 
        return [] 
    limit = image.shape[0] * 0.01 #testing instead of 0.02
    bboxes = [cv.boundingRect(c) for c in contours]
    bboxes = sorted(bboxes, key=lambda b: b[1])
    rows = []
    current_row = [bboxes[0]]

    for bbox in bboxes[1:]:
        if abs(bbox[1] - current_row[0][1]) < limit:
            current_row.append(bbox)
        else:
            rows.append(sorted(current_row, key=lambda z: z[0]))
            current_row = [bbox]

    rows.append(sorted(current_row, key=lambda q: q[0]))
    return [box for row in rows for box in row]


def splitter(image):
    file_bytes = np.frombuffer(image.read(), np.uint8)
    bnaimage = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
    bnaimage = cv.GaussianBlur(bnaimage, (5,5), 0) 
    thresh_image = cv.adaptiveThreshold(bnaimage, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)  #to match black on white 
    contours, _ = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_area = bnaimage.shape[0] * bnaimage.shape[1] 
    valid_contours = []
    for c in contours : 
        area = cv.contourArea(c)
        if( 0.06 * image_area) < area < (0.80 * image_area):
            valid_contours.append(c)
    
    panels_sorted = split_sort(valid_contours, bnaimage)
    panels = []
    padding = 3 #change as needed
    for x, y, w, h in panels_sorted:
        x1 = max(0, x- padding) 
        y1 = max(0, y - padding) 
        x2 = min(bnaimage.shape[1], x + w + padding) 
        y2 = min(bnaimage.shape[0], y+h+ padding) #for not going above the panel limits
        panel = bnaimage[y1:y2, x1:x2]         
        sub_panels = vertical_split_sort(panel, image_area, y1)
        if sub_panels is None : 
            panels.append(panel) 
        else  :
            panels.extend(sub_panels)

    return panels


def main():
    if "analysis_results" not in st.session_state : 
        st.session_state.analysis_results = {}
    
    st.set_page_config("wide") 
    st.title("Storyboard Analyzer &  Splitter")
    st.text("Upload an image to be split in different parts")
    model, processor = load_model() 

    text_original = st.text_area("Please input the text used to generate the storyboard for analysis.", placeholder = "Input the original text here." ) 

    text_input = f"""
    You are analysing sub-panels of a storyboard image. Each panel represents a part of the text used to generate the storyboard. Your task is to analyze the sub-panel and map it to the part of the text it corresponds to, and then describe the content of the panel in detail.
    
    Describe this image using the following specific categories : 
                    Object : descriptions of specific objects or entities.
                    Attribute : mentions of properties such as color, size, or shape.
                    Number : mentions of quantities or numerical values.
                    Text : descriptions of scene text or written content visible in the image.
                    Relation : semantic relationships of objects (e.g., prepositions or adjectives) within the description.
                    Fact : mentions of named entities such as people, places, or countries.
                    Scene_Description : an overall description of the image, describing the image as a whole and what it is showing.
                    Then, output the result in a JSON format using those categories only and nothing else.

Here is the text you ought to compare the sub panels to : {text_original}

Output the following : 

## STEP 1 - JSON ANALYSIS
Output ONLY this raw JSON, no backticks, no markdown:
{{
  "Object": [],
  "Attribute": {{}},
  "Number": {{}},
  "Text": "",
  "Relation": {{}},
  "Fact": {{}},
  "Scene_Description": ""
}}

## STEP 2 - TEXT ANNOTATION
Only if an original text is provided below, output the relevant portion 
with inline XML tags like <object>car</object>, <relation>beside</relation> etc.
If no text is provided output: "No initial text provided."

## ORIGINAL TEXT
{text_original if text_original else "None provided."}
"""
    
    user_text = st.text_area("Vision analysis instructions", value=text_input)
    file_uploaded = st.file_uploader("Upload a file", type=["png","jpg","jpeg"])

    if file_uploaded is not None:
        st.success("File uploaded successfully!")

        with st.spinner("Splitting..."):
            panels = splitter(file_uploaded)  # was outside if block, panels not captured
        
        st.text("Choose the panels which you want to keep and be analyzed by QWEN!")
        st.write("Please uncheck the panels you do not wish to keep.") 

        panels_kept = []

        for i, panel in enumerate(panels):
                print(type(panel), panel) 
                if isinstance(panel, tuple) : 
                    panel=panel[0]
                
                rgb_panel = cv.cvtColor(panel, cv.COLOR_GRAY2RGB)
                pil_panel = Image.fromarray(rgb_panel) 

                col1, col2 = st.columns([1,2]) 

                with col1: 
                    st.image(pil_panel, "Extracted panel")
                    panel_kept = st.checkbox(f"Panel number {i}", value = True, key = f"Panel_{i}" )

                    if panel_kept : 
                        panels_kept.append((i, pil_panel))

                with col2 : 
                    if panel_kept : 
                        if  i in st.session_state.analysis_results : 
                            st.markdown(st.session_state.analysis_results[i])
                        st.text("Ready for analysis!") 
                    else : 
                        st.text("Panel will not be analyzed.")

        if st.button("Run QWEN analsysis", type="secondary", key=f"Button_{i}") :
                for i, pil_panel in panels_kept : 
                    panel_result = run_qwen(pil_panel, processor, model, user_text) 
                    st.session_state.analysis_results[i] = panel_result 
                    st.success(f"Panel {i} has been sucessfully analyzed!")
                    st.markdown(panel_result) #for nice JSON formatting 
                    st.balloons() 
                
                
                    
    
        st.toast("Splitting the image has finished!")

if __name__ == "__main__":
    main()
