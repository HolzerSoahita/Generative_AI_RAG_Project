import os
import shutil
import tempfile
import uuid
import json
import gradio as gr
from huggingface_hub import HfApi

from library.chunking import (
    chunking_from_folder,
    create_db_from_document,
    create_db_from_text,
)
from library.FileSaver import FileSaver
from library.rag import Rag


def on_change_model(choice):
    """
        Gradio change model function
    """
    print(f"choice = {choice}")
    if choice == "HuggingFace":
        return gr.Group(visible=True),gr.Group(visible=False)
    elif choice == "From PC":
        return gr.Group(visible=False),gr.Group(visible=True)
    
def onChangeAlgoChoice(AlgoChoice):
    """
        Gradio algo choice function 
    """
    if AlgoChoice == "Agentic chunking":
        return gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=False)


def create_unique_userSubfolder():
    """
        function creation unique subfolder
    """

    # Generate a unique UUID
    unique_id = uuid.uuid4()
    user_unique_id.value = unique_id

    # Create the unique subfolder path
    subfolder_path = "database/user_" + str(unique_id)

    # Create the subfolder
    os.makedirs(subfolder_path, exist_ok=True)

    print(f"Created a new user subfolder with the name: {subfolder_path}")

    # Save the user folder path in the session variable
    user_folder.value = subfolder_path

    # Return the subfolder name
    return subfolder_path

def zip_all_use_cases_folder():
    """
        Function that zip all use case user folder 
    """

    # Define the directory name and zip file name
    directory_name = user_folder.value
    zip_folder = "zip/"+str(user_unique_id.value)+"/"
    zip_path = os.path.join(zip_folder, "database_vector_use_case_folder")

    # Create the directory if it doesn't exist
    os.makedirs(zip_folder, exist_ok=True)

    # Create a Zip file
    shutil.make_archive(zip_path, 'zip', directory_name)

def do_chunking(drp_uses_case_choice, path_uses_case_folder, drp_chunking_algo_choice, size_input, size_overlap, openai_api_key=None,progress=gr.Progress()):
    progress(0, "Starting")

    # User path
    path_user_folder = user_folder.value

    # If the user have not a subfolder
    progress(0.1, "Creation of the user subfolder")
    if path_user_folder=="":
        path_user_folder = create_unique_userSubfolder()
    
    print(f"Path user folder: {path_user_folder}")

    # Create the subfolder of the each use cases if doesn't exist
    for use_case in progress.tqdm(state_choice_use_cases.value, desc="Creating directories"):
        # Replace spaces with underscores and make the string lowercase
        folder_name = use_case.replace(" ", "_").lower()

        # Create the full path to the folder
        path = os.path.join(path_user_folder, folder_name)

        # Check if the folder already exists
        if not os.path.exists(path):
            # If not, create the folder
            os.makedirs(path)
            print(f"Directory '{path}' created")
        else:
            print(f"Directory '{path}' already exists")
    
    
    # Create a temporary directory for the uploaded files data from a directory
    progress(0.2, "Creation of the temporary folder")
    tmp_dir = tempfile.mkdtemp()

    # Copy each file to the new temporary directory
    for path in progress.tqdm(path_uses_case_folder, desc="Copying files"):
        shutil.copy(path, tmp_dir)

    # Chunking 
    progress(0.3, "Chunking...")
    if (drp_chunking_algo_choice == "Agentic chunking"):
        if (openai_api_key is None) or (openai_api_key.strip() == "") :
            return "Error : No Openai key",None
        else:
            try:
                data_chunk = chunking_from_folder(tmp_dir,size_input,size_overlap,type_chunking=drp_chunking_algo_choice,openai_api_key=openai_api_key)
            except:
                return "Error : Invalid Openai key",None
    else :
        data_chunk = chunking_from_folder(tmp_dir,size_input,size_overlap,type_chunking=drp_chunking_algo_choice)

    # Delete the temporary directory after chunking
    progress(0.6, "Removing temporary folder")
    shutil.rmtree(tmp_dir)

    # User use case folder
    use_case_folder_name = drp_uses_case_choice.replace(" ", "_").lower()
    # Create the full path to the user use case folder
    path_use_case_folder_name = os.path.join(path_user_folder, use_case_folder_name)

    # Save the chunking to the user use case folder
    progress(0.7, "Saving vector database...")
    if drp_chunking_algo_choice=="Agentic chunking":
        create_db_from_text(path_use_case_folder_name,data_chunk)
    else:
        create_db_from_document(path_use_case_folder_name,data_chunk)

    progress(0.9, "Creating the zip folder of all use cases...")
    zip_all_use_cases_folder()
    
    progress(1, "Chunking Finished")

    path_zip_folder_all_use_cases = "./zip/"+str(user_unique_id.value)+"/database_vector_use_case_folder.zip"
    
    return "Chunking Finished",path_zip_folder_all_use_cases


def test_connection_huggingface(hf_model_name, hf_api_key):
    """
        Function to test huggingface connexion
    """
    if (hf_api_key is None) or (hf_api_key.strip()==""):
        return "Hugging Face api invalid"
    
    try:
        # Create an instance of HfApi
        hf_api = HfApi(
            endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
            token=hf_api_key, # Token is not persisted on the machine.
        )

        # Check if the model exists
        model_info = hf_api.model_info(hf_model_name)

        if model_info:
            return "Connection to Hugging Face established successfully!!!"
        else:
            return f"Model '{hf_model_name}' does not exist on Hugging Face."
    except Exception as e:
        return f"Failed to establish connection to Hugging Face"


def do_parameterizing(choice_model,hf_model_name,hf_api_key,path_model,choice_resource,batch_size,max_seq_len,progress=gr.Progress()):
    """
        Function to do the gradio parameterizing
    """
    progress(0, "Starting")
    try:
        progress(0.1, "Parametizing...")
        if choice_model == "HuggingFace":
            user_rag_object.value = Rag(model=hf_model_name, model_type='llama', device=choice_resource.lower(), length=max_seq_len, creativity=0.1,batch_size=batch_size,hf_key=hf_api_key)
        elif choice_model == "From PC":
            user_rag_object.value = Rag(model=path_model, model_type='llama', device=choice_resource.lower(), length=max_seq_len, creativity=0.1,batch_size=batch_size)
        progress(0.9, "Parametizing finished...")
        return "Parameterizing successfull !!!"
    except:
        return "Parameterizing failed !!!"
    
def pretty_print_json_to_html(data):
    """
        Convert json data to html
    """
    # Convert string to dictionary
    if isinstance(data, str):
        data = json.loads(data.replace("'", "\""))

    print(f"Json data : {data}")
    
    # Increment the row number by 1 if it exists because it's begin by 0
    if 'row' in data:
        data['row'] = int(data['row']) + 1

    # Increment the page number by 1 if it exists because it's begin by 0
    if 'page' in data:
        data['page'] = int(data['page']) + 1

    # Convert each key-value pair to HTML unordered list
    html_code = "<ul>"
    for key, value in data.items():
        html_code += f"  <li>{key} : {value}</li>"
    html_code += "</ul>"
    
    return html_code

def do_RAG(path_vector_dataset_folder=None,area_prompt=None,progress=gr.Progress()):
    """
        Function to do the RAG
    """
    progress(0, "Starting")
    Rag_object = user_rag_object.value
    progress(0.1, "Vector configuration...")
    if path_vector_dataset_folder is not None:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Your code to use the temporary directory goes here
            # For example, if you want to copy all files from path_vector_dataset_folder to the temporary directory:
            for file_path in path_vector_dataset_folder:
                shutil.copy(file_path, temp_dir)

            # Use the temporary directory in place of add_local_db
            Rag_object.add_local_db(temp_dir)
            user_rag_object.value = Rag_object
        finally:
            # Ensure the temporary directory is deleted whether or not add_local_db() succeeds
            shutil.rmtree(temp_dir)
    
    progress(0.5, "RAG generation...")
    # doing rag
    if area_prompt is not None:
        try:
            output = Rag_object.ask(query=area_prompt)
        except:
            return "<p>Error system! Please, refresh the page and redo again</p>"
    else:
        return "Fill the prompt please!" 

    progress(0.9, "RAG completed! showing result...")
    if output is not None:
        result =f"""
        <h1>Result of the query : </h1>
        <p> {output["result"]} </p>
        <h1>Document informations : </h1>
        {pretty_print_json_to_html(output['source_documents'][0].metadata)}
        """
    else:
        result ="No database vector"
    progress(1, "Finished!!!")

    return result,result
    
def do_Augmented_RAG(prompt_creation,drp_model_available,creativity_level,tone,len_out,progress=gr.Progress()):
    """
        Function to do augmented RAG 
    """

    progress(0, "Starting")
    Rag_object = user_rag_object.value
    if drp_model_available =="LLama2":
        progress(0.1, "Applying modification...")
        model = "TheBloke/toxicqa-Llama2-7B-GGUF"
        Rag_object.augment(model_available=model,creativity_level = creativity_level,len_out=len_out)
        user_rag_object.value = Rag_object
        progress(0.5, "RAG generation...")
        if prompt_creation is not None:
            try:
                output = Rag_object.ask(query=prompt_creation,tone=tone)
            except:
                return "<p>Error system! Please, refresh the page and redo again</p>"
        else:
            return "Fill the prompt please!"
    elif drp_model_available =="Mistral":
        progress(0.1, "Applying modification...")
        model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        Rag_object.augment(model_available=model,creativity_level = creativity_level,len_out=len_out)
        user_rag_object.value = Rag_object
        progress(0.5, "RAG generation...")
        if prompt_creation is not None:
            try:
                output = Rag_object.ask(query=prompt_creation,tone=tone)
            except:
                return "<p>Error system! Please, refresh the page and redo again</p>"
        else:
            return "Fill the prompt please!"
    
    progress(0.9, "RAG completed! showing result...")
    if output is not None:
        result =f"""
        <h1>Result of the query : </h1>
        <p> {output["result"]} </p>
        <h1>Document informations : </h1>
        {pretty_print_json_to_html(output['source_documents'][0].metadata)}
        """
        save_final_state.value = result
    else:
        result ="No database vector"

    return result

def save_final(saving_option,progress=gr.Progress()):
    """
        Function to save the result of the augmented RAG
    """

    progress(0, "Starting")
    saver = FileSaver(html=save_final_state.value)

    progress(0.1, "Configuration...")
    # Define the directory name and zip file name
    directory_name = user_folder.value
    zip_folder = "zip/"+str(user_unique_id.value)+"/"

    # Create the directory if it doesn't exist
    os.makedirs(zip_folder, exist_ok=True)

    progress(0.3, "Saving...")
    final_file_path = ""
    if saving_option == 'DOCX':
        final_file_path = os.path.join(zip_folder, "final_file.docx")
        saver.save_docx(final_file_path)
    elif saving_option == 'PDF':
        final_file_path = os.path.join(zip_folder, "final_file.pdf")
        saver.save_pdf(final_file_path)
    elif saving_option == 'HTML':
        final_file_path = os.path.join(zip_folder, "final_file.html")
        saver.save_html(final_file_path)
    
    progress(1, "Finished...")
    
    return final_file_path



# Interface of Gradio
with gr.Blocks() as demo:
    state_choice_use_cases = gr.State(["Scientific papers","slides and technical reports","personal docs","Legal document Analysis"])

    gr.Markdown("RAG General platform")

    user_folder = gr.State("")
    user_unique_id = gr.State("")
    user_rag_object = gr.State(object)
    save_final_state = gr.State("")

    # The Tab Interface for initial chunking setup
    with gr.Tab("Init chunking setup") as Tab1:
        # file_dataset_extensions = ['.doc','.docx','.pptx','.xlsx','.odt','.csv','.txt','.md','.pdf','.json','.html']
        with gr.Group():
            gr.Markdown("Use-case Dataset")
            with gr.Row():
                drp_uses_case_choice= gr.Dropdown(choices=state_choice_use_cases.value,value="Scientific papers",interactive=True,label="Use case")
                path_uses_case_folder = gr.File(label="Folder path of the use case", type="filepath",file_count='directory')
        drp_chunking_algo_choice= gr.Dropdown(choices=["Fixed sized chunking", "Recursive chunking", "Content-aware chunking","Semantic chunking","Agentic chunking"],value="Fixed sized chunking",interactive=True,label="Chunking Algorithm")
        txt_openai_api_key = gr.Textbox(label="OpenAI API key", type="text",interactive=True,visible=False)
        drp_chunking_algo_choice.change(onChangeAlgoChoice,inputs=[drp_chunking_algo_choice],outputs=[txt_openai_api_key])
                
        with gr.Group():
            gr.Markdown("Chunking parameters")
            with gr.Row():
                size_input = gr.Number(value=256, label="Size input",interactive=True)
                size_overlap = gr.Number(value=20, label="overlap",interactive=True)
        
        button_chunck = gr.Button("Chunck")
        chunking_completed = gr.Markdown()

        button_chunck.click(do_chunking,inputs=[drp_uses_case_choice,path_uses_case_folder,drp_chunking_algo_choice,size_input,size_overlap,txt_openai_api_key],outputs=[chunking_completed,gr.File()])
        
        

    # The Tab Interface for Data selection and parameterization
    with gr.Tab("Data selection and parameterization") as Tab2:
        choice_model = gr.Radio(["HuggingFace", "From PC"], label="Choose the type of model",value="HuggingFace",interactive=True)
        with gr.Group(visible=(choice_model.value == "HuggingFace")) as hf_group:
            with gr.Row():
                hf_model_name = gr.Textbox(label="Model Name", type="text",value="TheBloke/Mistral-7B-v0.1-GGUF",interactive=True)
                hf_api_key = gr.Textbox(label="Hugging Face API key", type="text")
            button_connect_hf = gr.Button("Connect")
            test_connection_state = gr.Markdown()
            button_connect_hf.click(test_connection_huggingface,inputs=[hf_model_name,hf_api_key],outputs=[test_connection_state])
        
        with gr.Group(visible=(choice_model.value == "From PC")) as local_group:
            # path_model = gr.File(label="Path of the model", type="filepath",file_types=['.bin','.gguf'],file_count='directory')
            path_model = gr.File(label="Path of the model", type="filepath",file_types=['.bin','.gguf'],file_count='single')
        choice_resource = gr.Radio(["CPU", "GPU"], label="Choose the type of resource",value="CPU",interactive=True)
        
        with gr.Group():   
            with gr.Row():
                batch_size = gr.Number(value=8, label="Batch size",interactive=True)
                max_seq_len = gr.Number(value=256, label="Max sequence length",interactive=True)

        button_parameterize = gr.Button("Parameterize")
        button_parameterizing_completed = gr.Markdown()
        button_parameterize.click(do_parameterizing,inputs=[choice_model,hf_model_name,hf_api_key,path_model,choice_resource,batch_size,max_seq_len],outputs=[button_parameterizing_completed])

        choice_model.change(on_change_model, inputs=[choice_model],outputs=[hf_group,local_group])
        
    # The Tab Interface for doing RAG
    with gr.Tab("RAG") as Tab3:
        path_vector_dataset_folder = gr.File(label="Folder path of the vector dataset", type="filepath",file_count='directory')
        area_prompt = gr.TextArea(label="Prompt", type="text",interactive=True)
        start_rag = gr.Button("Start RAG")
        html_generated_answer_rag= gr.HTML()
        
    # The Tab Interface for augmenting Context
    with gr.Tab("Augmenting Context") as Tab4:
        retreived_context = gr.HTML()
        prompt_creation = gr.TextArea(label="Prompt", type="text",interactive=True)
        examples = [
            "Write a comprehensive blog post summarizing these findings.",
            "Generate an academic paper abstract based on this context.",
            "Create a detailed report comparing these use cases.",
            "Formulate a new research question inspired by this context.",
            "Reformulate this context for a layman audience."
        ]
        example_component = gr.Examples(examples, prompt_creation,label="Prompt example")
        with gr.Group():   
            gr.Markdown("LLM Selection and Configuration")
            with gr.Row():
                drp_model_available= gr.Dropdown(choices=["LLama2", "Mistral"],value="Mistral",interactive=True,label="Model available")
                creativity_level = gr.Slider(minimum=0, maximum=1, value=0.5, label="Creativity level",interactive=True)
                list_tones = ["Formal", "Informal", "Serious", "Humorous", "Sarcastic", "Optimistic", "Pessimistic", "Joyful", "Sad", "Respectful", "Irreverent", "Sympathetic", "Empathetic", "Angry", "Calm", "Passionate", "Objective", "Subjective", "Critical", "Laudatory", "Energetic", "Lazy", "Enthusiastic", "Indifferent", "Doubtful", "Confident", "Fearful", "Hopeful", "Despairing", "Loving", "Hateful", "Excited", "Bored", "Surprised", "Disgusted", "Admiring", "Disapproving", "Proud", "Ashamed", "Jealous", "Apathetic", "Curious"]
                tone = gr.Dropdown(choices=list_tones,value="Formal",interactive=True,label="Tone")
                len_out = gr.Number(value=256, label="Length of output",interactive=True)
        
        button_generate = gr.Button("Generate")
        html_generated_answer_rag_augmenting_context= gr.HTML()
        button_generate.click(do_Augmented_RAG,inputs=[prompt_creation,drp_model_available,creativity_level,tone,len_out],outputs=[html_generated_answer_rag_augmenting_context])
       
        list_saving_option = ['DOCX', 'PDF', 'HTML']
        saving_option = gr.Dropdown(choices=list_saving_option,value='HTML',interactive=True,label="Saving option")
        button_save = gr.Button("Save")
        button_save.click(save_final,inputs=[saving_option],outputs=gr.File())

    start_rag.click(do_RAG,inputs=[path_vector_dataset_folder,area_prompt],outputs=[html_generated_answer_rag,retreived_context])


if __name__ == "__main__":
    # Launching the gradio interface
    demo.launch(debug=True)




        

