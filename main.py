import gradio as gr
from library.chunking import chunking_from_folder,create_db_from_document,create_db_from_text
from library.rag import Rag
import os, uuid, shutil, tempfile

def on_change_model(choice):
    print(f"choice = {choice}")
    if choice == "HuggingFace":
        return gr.Group(visible=True),gr.Group(visible=False)
    elif choice == "From PC":
        return gr.Group(visible=False),gr.Group(visible=True)
    
def onChangeAlgoChoice(AlgoChoice):
    if AlgoChoice == "Agentic chunking":
        return gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=False)


def create_unique_userSubfolder():
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



# Interface of Gradio
with gr.Blocks() as demo:
    state_choice_use_cases = gr.State(["Scientific papers","slides and technical reports","personal docs","Legal document Analysis"])

    gr.Markdown("RAG General platform")

    user_folder = gr.State("")
    user_unique_id = gr.State("")

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
                api_key = gr.Textbox(label="API key", type="text")
            button_connect_hf = gr.Button("Connect")
        with gr.Group(visible=(choice_model.value == "From PC")) as local_group:
            # path_model = gr.File(label="Path of the model", type="filepath",file_types=['.bin','.gguf'],file_count='directory')
            path_model = gr.File(label="Path of the model", type="filepath",file_count='directory')
        choice_resource = gr.Radio(["CPU", "GPU"], label="Choose the type of resource",value="CPU",interactive=True)
        with gr.Group():   
            with gr.Row():
                batch_size = gr.Number(value=64, label="Batch size",interactive=True)
                max_seq_len = gr.Number(value=64, label="Max sequence length",interactive=True)

        button_parameterize = gr.Button("Parameterize")

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
       
        list_saving_otion = ['DOCX', 'PDF', 'HTML']
        saving_option = gr.Dropdown(choices=list_saving_otion,value='PDF',interactive=True,label="Saving option")
        button_save = gr.Button("Save")

if __name__ == "__main__":
    demo.launch(debug=True)




        

