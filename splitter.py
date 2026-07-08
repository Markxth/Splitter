import streamlit as st
from PIL import Image
from time import sleep
import cv2 as cv
import numpy as np 
from dotenv import find_dotenv, load_dotenv
import torch 
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from google import genai
import os
import json 
from time import sleep
from io import BytesIO
import base64
from streamlit_cropper import st_cropper 
from google.genai import types
from summac.model_summac import SummaCConv #the authors reccomend this one instead of SummaCZS
from streamlit_sortables import sort_items

load_dotenv(find_dotenv()) 
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) #Gemini
#replace this as needed.

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
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        
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
    if not contours : 
        return [] 
    limit = image.shape[0] * 0.01 #0.01 can be changed on a per-image basis.
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

def text_analysis(text_original, text_analysis_instructions) :
    """ IF working with a Hugging Face model uncomment this and modift the arguments to process the model, the processor and the input text
    messages = [
    {
        "role": "system",
        "content": text_analyis_instructions
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text_input
            }
        ]
    }
] 
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) 
    inputs = processor(
    text=[text],
    padding=True,
    return_tensors="pt",
)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    return output_text[0] 
    """
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=text_original,
        config = {
            "system_instruction" : text_analysis_instructions
        }
    )
    output = response.text
    return output 

def to_b64(img) : 
    if img is None :
        st.warning("No image was passed!") 
        return ""

    buffer = BytesIO() 
    img.save(buffer, format = "PNG") 
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")
    
def gemini_analysis(session_panels) : #embed text being the analysis results, idk why I named it that way
    output = {} #for storing the results and easily working with them
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for i in session_panels:
        idx, img = i
        # Convert PIL image (or path) into bytes
        if isinstance(img, str):
            image = Image.open(img)
        else:
            image = img

        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        prompt = f"""You are given a panel. For this panel following the following instructions : 

        Text : For the panel create a phrase describing the panel, specifically, what is happening in the image.
"""

        response = client.models.generate_content (
            model="gemini-2.5-flash",
            contents=[
                prompt,
                    types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png",
        ),
    ],
        )
        print(response.text)
        output[idx] = response.text 

    return output

def textcomp(original_text, gemini_results) : 

    responses = []

    for i, description in gemini_results.items() : 
        
        instruction = f"""Here is the original text : {original_text} 
        
        Here are the analysis results : {description}

        The original text represents the premise, being the original text used to generate the storyboard. The analysis results are an AI-generat description of a subpanel of the storyboard.

        Treating the original text as the premise, evaluate how the 2 texts match using the following criteria : 

        Match - if the text follows logically as part of the original text, and is consistent with the original text.
        No match - if the text does not follow logically as part of the original text, and is inconsistent with the original text.
        Neutral - if the text is neither consistent nor inconsistent with the original text, but rather in between with SOME parts being consistent and some not.
        """
        print(f"Analysis for panel number {i} : ") 

        response = client.models.generate_content (
                model="gemini-2.5-flash",
                contents=[
                    instruction,
        ],
            )
        print(response.text)
        responses.append(response.text) 
    return responses

def summac(original_text, gemini_results) : 
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean") 

    initial_text = original_text
    for i, description in gemini_results.items() : 

        gemini_text = description  

    summac_score = model_conv.score([initial_text], [description]) #unpack gemini results
    return summac_score 

def main():
    #due to the way Streamlit works, one wants to store everything per run in a cache, else all is lost per rerun.
    if "text_results" not in st.session_state : 
        st.session_state.text_results = [] 
    if "panels_kept" not in st.session_state : 
        st.session_state.panels_kept = [] 
    if "cropped_panels" not in st.session_state : 
        st.session_state.cropped_panels = {} #PIL image
    if "crop_mode" not in st.session_state :
        st.session_state.crop_mode = {} 
    if "analysis_results" not in st.session_state : 
        st.session_state.analysis_results = {} 
    if "trash_bin" not in st.session_state : 
        st.session_state.trash_bin = {} 
    if "gem_results" not in st.session_state :
        st.session_state.gem_results= {} 
    if "textcomp" not in st.session_state :
        st.session_state.textcomp = {}
    if "summac_results" not in st.session_state :
        st.session_state.summac_results = [] 
    if "manual_crops" not in st.session_state :
        st.session_state.manual_crops = {} 
    if "manual_crop_mode" not in st.session_state : 
        st.session_state. manual_crop_mode=  {} 
    if "panels_order" not in st.session_state: 
        st.session_state.panels_order = []

    st.set_page_config("Splitter & Storyboard Analyzer", layout = "wide") 
    st.title("Storyboard Analyzer &  Splitter")
    st.text("Upload an image to be split in different parts")
    model, processor = load_model() 

    text_original = st.text_area("Original vignette text.", placeholder = "Paste the original vignette or scenario here." ) 

#QWEN instructions
    text_input = f"""You are analysing a single sub-panel of a storyboard image.
 
## STEP 1 - VISUAL ANALYSIS (image only)
Look ONLY at what is physically visible in the image. Do not use the reference text below.
Output raw valid JSON with no backticks, no markdown and nothing extra added to it. Only the JSON results:
{{
  "Object": [],
  "Attribute": {{}},
  "Number": {{}},
  "Text": "",
  "Relation": {{}},
  "Fact": {{}},
  "Scene_Description": "",
  
Categories:
-

After this at another section named "Justifications" where you justify and argue why you gave the answers you gave in relation to the Scene_Description category IN DETAIL. 
    
DO NOT SKIP JUSTIFICATIONS. If you believe you cannot provide a justification, say WHY.    
"""

#text embedding 
    text_analyis_instructions = f"""Text_Mapping": {{
    "Original text : {text_original}
    "Embedded_Matching_Text": "Rewrite the original text with inline XML tags marking entities using the attributes in {text_input} (e.g., <object>car</object>, <relation>beside</relation>, <attribute>red</attribute>). Do NOT add or change the text in ANY other way than indicated."
    }}
"""  
    
    user_text = st.text_area("Vision analysis instructions", value=text_input)
    file_uploaded = st.file_uploader("Upload a file", type=["png","jpg","jpeg"], accept_multiple_files = True)
    if file_uploaded:
        panels = [] 
        st.success("File uploaded successfully!")

        with st.spinner("Splitting..."):
            for img_idx in range(len(file_uploaded)) : 
                file = file_uploaded[img_idx]
                panels_uploaded = splitter(file) 
                for panel in panels_uploaded :
                    panels.append((img_idx, panel)) 


        st.text("Choose the panels which you want to keep and be analyzed by QWEN.")
        st.write("Please uncheck the panels you do not wish to keep.") 

        panels_kept = []

        #manual crop section 
        for img_idx, file in enumerate(file_uploaded) : #go through all files and put the counter back in the beginnig of the file, as .read() moves it to the end
            file.seek(0)
            original_image = Image.open(file).convert("RGB")  
            st.image(original_image, caption  = "Original uploaded file")
            
            if st.session_state.manual_crop_mode.get(img_idx, False) :
                preview = st_cropper(
                        original_image,
                        realtime_update = True, 
                        aspect_ratio = None, 
                        key=f"manual_crops_{img_idx}" 
                    )
                c1_1, c2_1 = st.columns(2)
            
                with c1_1 : 
                        if st.button("Save crop", key = f"save_manual_crop_{img_idx}") : 
                            manual_id_cropped = f"{img_idx}_manual_crop_{len(st.session_state.manual_crops)}"
                            st.session_state.manual_crops[manual_id_cropped] = preview
                            st.session_state.manual_crop_mode[img_idx] = False
                            st.rerun() 
                            
                with c2_1 : 
                    
                        if st.button("Cancel crop", key = f"cancel_manual_crop_{img_idx}") : 
                            st.session_state.manual_crop_mode[img_idx] = False
                            st.rerun()
            
            else: 
                manual_crop_coloumn ,reset_manual_crop_coloumn = st.columns(2)
                with manual_crop_coloumn : 
                    if st.button("Crop image", key  =f"save_crop_true_{img_idx}") :
                        st.session_state.manual_crop_mode[img_idx]= True
                        st.rerun()  
                with reset_manual_crop_coloumn :
                    if img_idx in st.session_state.manual_crops : 
                        if st.button("Cancel crop" , key =f"canceled_crop_{img_idx}") : 
                            del st.session_state.manual_crops[img_idx] 
                            st.rerun() 

        #automatic crop section
        with st.expander("Extracted panels", expanded  = True) :
            if st.session_state.manual_crops is not None : 
                for panel_id, pil_panel in st.session_state.manual_crops.items() : 
                    st.image(pil_panel, f"Manual crop {panel_id}")
                    width, height = pil_panel.size
                    st.text(f"Panel size : {width} x {height} pixels")
                    if st.button("Delete manual crop", key = f"deleted_manual_crop_{panel_id}") : 
                        st.session_state.trash_bin[panel_id] = {
                            "img"  : st.session_state.manual_crops.get(panel_id, pil_panel),
                            "analysis" : st.session_state.analysis_results.get(panel_id)
                        } 
                        del st.session_state.manual_crops[panel_id] 
                        st.session_state.manual_crops.pop(panel_id, None) 
                        st.session_state.analysis_results.pop(panel_id, None)
                        st.rerun()

            for panel_idx, (img_idx, panel) in enumerate(panels): 
                panel_id = f"{img_idx}_{panel_idx}" 
                if panel_id in st.session_state.trash_bin : 
                    continue 
                print(type(panel), panel) 
                if isinstance(panel, tuple) : 
                    panel=panel[0]
                
                rgb_panel = cv.cvtColor(panel, cv.COLOR_GRAY2RGB)
                pil_panel = Image.fromarray(rgb_panel) 
                pil_panel_crop = st.session_state.cropped_panels.get(panel_id, pil_panel) 

                col1, col2 = st.columns([1,2]) 

                with col1: 
                    if st.session_state.crop_mode.get(panel_id, False) :
                        preview = st_cropper(
                            pil_panel,
                            realtime_update = True, 
                            aspect_ratio = None, 
                            key=f"cropper_{panel_id}" 
                        )

                        c1,c2 = st.columns(2) 
                        with c1 :
                            if st.button("Save crop", key  =f"save_crop_false_{panel_id}") :
                                st.session_state.cropped_panels[panel_id] = preview
                                st.session_state.crop_mode[panel_id]= False 
                                st.rerun()  
                        with c2 :
                            if st.button("Cancel crop" , key =f"canceled_crop_{panel_id}") : 
                                st.session_state.crop_mode[panel_id]= False
                                st.rerun() 
                    else: 
                        st.image(pil_panel_crop, "Extracted panel")
                        width, height = pil_panel_crop.size
                        st.text(f"Panel size : {width} x {height} pixels")
                        crop_coloumn ,reset_crop_coloumn = st.columns(2)
                        with crop_coloumn : 
                            if st.button("Crop image", key  =f"save_crop_true_{panel_id}") :
                                st.session_state.crop_mode[panel_id]= True
                                st.rerun()  
                        with reset_crop_coloumn :
                            if panel_id in st.session_state.cropped_panels : 
                                if st.button("Cancel crop" , key =f"canceled_crop_{panel_id}") : 
                                    del st.session_state.cropped_panels[panel_id] 
                                    st.rerun() 
                    
                    if st.button("Delete panel", key = f"deleted_panel_{panel_id}") : 
                        st.session_state.trash_bin[panel_id] = {
                            "img" : st.session_state.cropped_panels.get(panel_id, pil_panel_crop),
                            "analysis" : st.session_state.analysis_results.get(panel_id)
                        }
                        st.session_state.cropped_panels.pop(panel_id, None) #none for error handling, though one should not be able to delete something non-existing.
                        st.session_state.analysis_results.pop(panel_id, None) 
                        st.rerun() 
                    
                    
                    panel_kept = st.checkbox(f"Panel number {panel_id}", value = True, key = f"Panel_{panel_id}" )
                    if panel_kept : 
                        panels_kept.append((panel_id, pil_panel_crop))
                        for panel_id in st.session_state.manual_crops :
                            panels_kept.append(st.session_state.manual_crops.get(panel_id)) #add the manual crops to the panels kept

                with col2 : 
                    if panel_kept : 
                        if  panel_id in st.session_state.analysis_results : 
                            st.markdown(st.session_state.analysis_results[panel_id])
                        st.text("Ready for analysis!") 
                    else : 
                        st.text("Panel will not be analyzed.")
                    
        if st.session_state.trash_bin is not None : 
            with st.expander("Trash bin"): 
                for panel_id, trash_item in list(st.session_state.trash_bin.items()) : 
                    st.image(trash_item["img"], f"Analysis : {trash_item['analysis']}") 
                    if st.button("Restore panel" , key = f"restored_panle_{panel_id}") : 
                        if panel_id in st.session_state.trash_bin : 
                            trash_item = st.session_state.trash_bin[panel_id]
                            st.session_state.cropped_panels[panel_id] = trash_item["img"] 
                            if trash_item["analysis"] is not None :
                                st.session_state.analysis_results[panel_id] = trash_item["analysis"] 
                            del st.session_state.trash_bin[panel_id] 
                            st.rerun() 
            
        
        if st.button("Run QWEN analsysis", type="secondary", key="run_qwen_analysis") :
            for panel_id, pil_panel in panels_kept : 
                panel_result = run_qwen(pil_panel, processor, model, user_text) 
                if isinstance(panel_result, list) : 
                    panel_result = panel_result[0] 
                panel_result = panel_result.strip().removeprefix("json").strip()
                st.session_state.analysis_results[panel_id] = panel_result 
                st.success(f"Panel {panel_id} has been sucessfully analyzed!")
                st.code(panel_result, language = 'json') #for nice JSON formatting 
            st.session_state.panels_kept = panels_kept 

        if st.button("Export to JSON") :
            if not st.session_state.panels_kept : 
                st.warning("Run the analysis first!")
            with open("results.json" , "w") as f : 
                json.dump(
                    st.session_state.analysis_results, f, indent = 4
                )
            st.success("Succesfully exported to a JSON file!") 
    
    
    if st.button("Embed the text") :             
        text_analysis_result = text_analysis(text_original, text_analyis_instructions)
        st.session_state.text_results = text_analysis_result
        st.markdown(text_analysis_result)


    if st.button("Generate Panel Descriptions") : 
        with st.expander("Generated panel descriptions", expanded = True) : 
            if not st.session_state.analysis_results:
                st.warning("Run the QWEN analysis first!")
            else : 
                generated_descriptions = gemini_analysis(st.session_state.panels_kept)
                st.markdown("\n".join(generated_descriptions.values()) )
                st.session_state.gem_results.update(generated_descriptions) 
                print(st.session_state.analysis_results) 

    if st.button("Run LLMaaJ entailment analysis") : 
        if not st.session_state.gem_results : 
            st.warning("Run the text analysis first!") 
        else : 
            st.session_state.textcomp = textcomp(text_original, st.session_state.gem_results) 
            st.session_state.textcomp.update({i: instruction for i, instruction in st.session_state.gem_results.items()})

        if st.session_state.textcomp:
            st.markdown(" ".join(st.session_state.textcomp))  

    if st.button("Run SummaC Analysis") :
        if not st.session_state.gem_results : 
            st.warning("Generate a panel description first.")
        else :
            analysis = summac(text_original, st.session_state.gem_results) 
            st.session_state.summac_results.append(analysis)

if __name__ == "__main__":
    main()
