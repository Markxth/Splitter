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
import subprocess 


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
        model="gemini-2.5-flash", 
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
    
def gemini_analysis(session_panels) : 
    output = {} #for storing the results and easily working with them
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for panel_id, img in session_panels:
        # Convert PIL image (or path) into bytes
        if isinstance(img, str):
            image = Image.open(img)
        else:
            image = img

        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        prompt = f"""You are given a panel. For this panel following the following instructions : 

        Text : For the panel create a phrase describing the panel, specifically, what is happening in the image."""

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
        output[panel_id] = response.text 

    return output

def textcomp(original_text, gemini_results) : 
    responses = {}
    for panel_id, description in gemini_results.items() : 
        
        instruction = f"""Here is the original text : {original_text} 
        
        Here are the analysis results : {description}

        The original text represents the premise, being the original text used to generate the storyboard. The analysis results are an AI-generat description of a subpanel of the storyboard.

        IMPORTANT: BEFORE judging, check the original text for any caveats, exceptions, or conditional statements (e.g. "sometimes X happens," "in some cases," "not always", "sometimes bad, sometimes good"). If the original text explicitly allows for variation or failure modes, a panel depicting that variation or failure should NOT be judged as a mismatch, even if it doesn't match the primary/happy-path description, but rather as a match, as it helps reach the final goal.

        Treating the original text as the premise, evaluate how the 2 texts match using the following criteria : 

        Match - if the text follows logically as part of the original text, and is consistent with the original text.
        No match - if the text does not follow logically as part of the original text, and is inconsistent with the original text.
        Neutral - if the text is neither consistent nor inconsistent with the original text, but rather in between with SOME parts being consistent and some not.
        """
        print(f"Analysis for panel number {panel_id} : ") 

        response = client.models.generate_content (
                model="gemini-2.5-flash",
                contents=[
                    instruction,
        ],
            )
        print(response.text)
        responses[panel_id] = response.text
    return responses

def summac(original_text, gemini_results) : 
    #print("[DEBUG] SummaC button clicked, gem_results has", len(st.session_state.gem_results), "entries")
    scores =  {} 
    for panel_id, description in gemini_results.items() : 
        payload = json.dumps({"original_text" : original_text, "description" : description})
        try : 
            result = subprocess.run(
                [r"C:\Users\markt\scoop\apps\miniconda3\current\envs\summac_env\python.exe",r"C:\Users\markt\coint\sumc.py"],
                input=payload, #stdin 
                capture_output = True, #so stdout is in the variable we control and not directly printed to terminal
                text= True,
            # timeout = 90 #give the NLI model time to load
            )
        
        except subprocess.TimeoutExpired :
            print("Timeout for panel : {panel_id} ")
            scores[panel_id] = None
            continue
        stdout = result.stdout
        stderr = result.stderr 
        if result.returncode != 0 : 
            print("SummaC had errors!")
            scores[panel_id] = None
            continue
        #the output format here is this way to avoid the errors of stdout processing all the info
        for line in stdout.splitlines() :
            if line.startswith("SummaC result"):
                output = json.loads(line[len("SummaC result") :] )
        scores[panel_id] = output["score"] #the variable within  output[] must match the name from the summac.py file
    return scores 

def hallucination(llmaaj, summac, qwen, original_text, gemini_desc, storyboard_panels, panel_id):
    """
    storyboard_panels: dict of {panel_id: PIL image} — the full sequence, for narrative context
    panel_id: the ID of the panel currently being judged, so we know which one to flag as the subject
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def to_bytes(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    ordered_ids = sorted(storyboard_panels.keys())
    sequence_parts = []
    for pid in ordered_ids:
        marker = f"[PANEL {pid}]" + ("is the panel being judged currently." if pid == panel_id else "")
        sequence_parts.append(marker)
        sequence_parts.append(types.Part.from_bytes(data=to_bytes(storyboard_panels[pid]), mime_type="image/png"))

    prompt = f"""You are a hallucination detector for one specific panel within a full storyboard sequence.

You are given the FULL storyboard (all panels, in order) for narrative context, plus these signals specifically about the panel marked "THIS IS THE PANEL BEING JUDGED":

Original text : {original_text if original_text else "No original text has been provided, the analysis will go forward without it."}
Panel description : {gemini_desc if gemini_desc else "No panel description has been provided, the analysis will go forward without it."}
QWEN analysis : {qwen if qwen else "No QWEN analysis has been provided, the analysis will go forward without it."}
LLM-as-a-Judge entailment verdict : {llmaaj if llmaaj else "NO LLMaaJ analysis has been provided, the analysis will go forward without it."}
SummaC score : {summac if summac else "No SummaC analysis has been provided, the analysis will go forward without it."}

Using the full storyboard sequence as narrative context, judge ONLY the marked panel. A panel that deviates from the "happy path" description in the original text is NOT automatically a hallucination if the original text allows for exceptions, variation, or alternate outcomes — check the full original text carefully for such caveats before flagging.

If there are elements which have not been provided mention that you are formulating your answer with the given details and ALWAYS that it is worth double checking since you were not given a full context.

Return ONLY valid JSON, no markdown, no backticks, in this exact shape:
{{
  "issues": [
    {{
      "category": "Object" | "Attribute" | "Relation" | "Text" | "Scene_Description" | "Other",
      "title": "short 5-8 word summary",
      "explanation": "why this was flagged, referencing the image and/or narrative context",
      "confidence": 0.0 to 1.0
    }}
  ]
}}

If there are no hallucinations, return {{"issues": []}}.
"""

    contents = [prompt] + sequence_parts

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("issues", [])
    except json.JSONDecodeError:
        return [{"category": "Other", "title": "Parse error", "explanation": raw, "confidence": None}]

def main():
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
        st.session_state.summac_results = {}  
    if "manual_crops" not in st.session_state :
        st.session_state.manual_crops = {} 
    if "manual_crop_mode" not in st.session_state : 
        st.session_state. manual_crop_mode=  {} 
    if "panels_order" not in st.session_state: 
        st.session_state.panels_order = []
    if "hallucinations" not in st.session_state : 
        st.session_state.hallucinations = {} 
    if "manual_notes" not in st.session_state : 
        st.session_state.manual_notes = {} 
    if "editing_notes" not in st.session_state : #automated hallucination = bool 
        st.session_state.editing_notes= {} 
    if "hallucination_out" not in st.session_state :  #for saving automated 
        st.session_state.hallucination_out  = {} 
    if "manual_note_edit" not in st.session_state : #manual hallucination   
        st.session_state.manual_note_edit = {}
    if "editing_manual" not in st.session_state: #for manual hallucination editing - bool 
        st.session_state.editing_manual = {} 
        
    st.set_page_config("Splitter & Storyboard Analyzer", layout = "wide") 
    st.title("Storyboard Analyzer &  Splitter")
    st.text("Upload an image to be split in different parts")
    model, processor = load_model() 

    # [ERROR FIX]: Moved the Configuration Panel BEFORE the instruction strings.
    # We must define 'text_original' and 'user_text' first so the f-strings below can access them.
    with st.sidebar:
        st.header("Configuration Panel")
        
        # 1. Define text_original first
        text_original = st.text_area("Original vignette text.", placeholder = "Paste the original vignette or scenario here." ) 
        
        # 2. Define the default JSON instructions
        text_input_default = f"""You are analysing a sub-panel of a storyboard image.
 
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
  "Scene_Description": ""
}}
    
DO NOT SKIP JUSTIFICATIONS. If you believe you cannot provide a justification, say WHY.    
"""
        # 3. Create the text area using the default
        user_text = st.text_area("Vision analysis instructions", value=text_input_default)
        
        # 4. File uploader
        file_uploaded = st.file_uploader("Upload a file", type=["png","jpg","jpeg"], accept_multiple_files = True)


    text_analyis_instructions = f"""Text_Mapping": {{
    "Original text : {text_original}
    "Embedded_Matching_Text": "Rewrite the original text with inline XML tags marking entities using the attributes in {user_text} (e.g., <object>car</object>, <relation>beside</relation>, <attribute>red</attribute>). Do NOT add or change the text in ANY other way than indicated."
    }}
""" 
    
    if file_uploaded:
        panels = [] 
        st.success("File uploaded successfully!")

        with st.spinner("Splitting..."):
            for img_idx in range(len(file_uploaded)) : 
                file = file_uploaded[img_idx]
                panels_uploaded = splitter(file) #becasue this thing returns i can then use it right below
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
                    
                with col2 : 
                    if panel_kept: 
                        if  panel_id in st.session_state.analysis_results : 
                            st.markdown(st.session_state.analysis_results[panel_id]) 
                        st.text("Ready for analysis!") 
                    else : 
                        st.text("Panel will not be analyzed.")
                        
            for man_id, man_img in st.session_state.manual_crops.items() :
                panels_kept.append((man_id, man_img)) #add the manual crops to the panels kept

            st.session_state.panels_kept = panels_kept
                
        if st.session_state.trash_bin is not None : 
            with st.expander("Trash bin"): 
                for panel_id, trash_item in list(st.session_state.trash_bin.items()) : 
                    st.image(trash_item["img"], f"Analysis : {trash_item['analysis']}") 
                    if st.button("Restore panel" , key = f"restored_panel_{panel_id}") : 
                        if panel_id in st.session_state.trash_bin : 
                            trash_item = st.session_state.trash_bin[panel_id]
                            st.session_state.cropped_panels[panel_id] = trash_item["img"] 
                            if trash_item["analysis"] is not None :
                                st.session_state.analysis_results[panel_id] = trash_item["analysis"] 
                            del st.session_state.trash_bin[panel_id] 
                            st.rerun() 
            
        
        with st.expander("Actions & Analysis Hub", expanded=True):
            if st.button("Run QWEN analsysis", type="secondary", key="run_qwen_analysis") :
                st.spinner("Analyzing panels...")
                counter = 0
                for panel_id, pil_panel in panels_kept : 
                    counter+=1 
                    panel_result = run_qwen(pil_panel, processor, model, user_text) 
                    if isinstance(panel_result, list) : 
                        panel_result = panel_result[0] 
                    description = panel_result.strip()
                    description = description.removeprefix("```json").removeprefix("```").strip()
                    description = description.removesuffix("```").strip()

                    st.session_state.analysis_results[panel_id] = description 
                    # [TYPO FIX]: Fixed typo in success message from "sucessfully" to "successfully".
                    st.success(f"Panel {panel_id} has been successfully analyzed!")
                    st.code(panel_result, language = 'json') #for nice JSON formatting 
                st.session_state.panels_kept = panels_kept 
                # [TYPO FIX]: Fixed typo in text message from "panals" to "panels".
                st.text(f"Number of panels analyzed : {counter}")

            if st.button("Export to JSON") :
                if not st.session_state.panels_kept : 
                    st.warning("Run the analysis first!")
                else :
                    with open("results.json" , "w") as f : 
                        json.dump(
                            st.session_state.analysis_results, f, indent = 4   
                        )
                st.success("Successfully exported to a JSON file!") 
        
        
            if st.button("Embed the text") :             
                text_analysis_result = text_analysis(text_original, text_analyis_instructions)
                st.session_state.text_results = text_analysis_result
                st.markdown(text_analysis_result)


            if st.button("Generate Panel Descriptions") : 
                generated_descriptions = gemini_analysis(panels_kept)
                st.session_state.gem_results.update(generated_descriptions)
                
                with st.expander("Generated panel descriptions", expanded = True) : 
                    generated_descriptions = gemini_analysis(panels_kept)
                    st.markdown("\n\n".join(generated_descriptions.values()) ) #markdown requies \n\n
                    st.session_state.gem_results.update(generated_descriptions) 
                    print(st.session_state.analysis_results) 

            if st.button("Run LLMaaJ entailment analysis") : 
                if not st.session_state.gem_results : 
                    st.warning("Run the text analysis first!")  
                else : 
                    st.session_state.textcomp = textcomp(text_original, st.session_state.gem_results) 
                    
                if st.session_state.textcomp:
                    for panel_id, verdict in st.session_state.textcomp.items() : 
                        st.markdown(f"Panel : {panel_id}  : {verdict}")
                    
            if st.button("Run SummaC Analysis") :
                if not st.session_state.gem_results : 
                    st.warning("Generate a panel description first.")
                else :
                    analysis = summac(text_original, st.session_state.gem_results) 
                    st.session_state.summac_results = analysis #summac already returns a dictionary
                    st.write(st.session_state.summac_results)
                if st.session_state.summac_results :
                    for panel_id , description in st.session_state.summac_results.items() :
                        print(f"SummaC analysis for panel {panel_id} : {description}")
                
            if st.button("Check hallucinations") : 
                    for panel_id, panel_img in panels_kept:
                        qwen_output = st.session_state.analysis_results.get(panel_id)
                        gemini_description = st.session_state.gem_results.get(panel_id)
                        entailment_verdict = st.session_state.textcomp.get(panel_id)
                        summac_score = st.session_state.summac_results.get(panel_id)

                        result = hallucination(
                            entailment_verdict,
                            summac_score,
                            qwen_output,
                            text_original,
                            gemini_description,
                            {pid: img for pid, img in panels_kept},
                            panel_id
                        )
                        st.session_state.hallucination_out[panel_id] = result

            if st.session_state.hallucination_out:
                with st.expander("Hallucination Audit Results", expanded=True):
                    for panel_id, issues in st.session_state.hallucination_out.items():
                        st.markdown(f"**Panel {panel_id} Audit:**")
                        if not issues:
                            st.success("No issues found!")
                        else:
                            for issue in issues:
                                st.warning(f"[{issue.get('category')}] {issue.get('title')}: {issue.get('explanation')}")

if __name__ == "__main__":
    main()