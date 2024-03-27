import gradio as gr

def on_change_model(choice):
    print(f"choice = {choice}")
    if choice == "HuggingFace":
        return gr.Group(visible=True),gr.Group(visible=False)
    elif choice == "From PC":
        return gr.Group(visible=False),gr.Group(visible=True)

# Interface of Gradio
with gr.Blocks() as demo:
    state_choice_use_cases = gr.State(["Scientific papers","slides and technical reports","personal docs","Legal document Analysis"])

    gr.Markdown("RAG General platform")

    # The Tab Interface for initial chunking setup
    with gr.Tab("Init chunking setup") as Tab1:
        # file_dataset_extensions = ['.doc','.docx','.pptx','.xlsx','.odt','.csv','.txt','.md','.pdf','.json','.html']
        with gr.Group():
            gr.Markdown("Use-case Dataset")
            with gr.Row():
                drp_uses_case_choice= gr.Dropdown(choices=state_choice_use_cases.value,value="Scientific papers",interactive=True,label="Use case")
                # path_uses_case_folder = gr.File(label="Folder path of the use case", type="filepath",file_types=file_dataset_extensions,file_count='directory')
                path_uses_case_folder = gr.File(label="Folder path of the use case", type="filepath",file_count='directory')
        drp_chunking_algo_choice= gr.Dropdown(choices=["Fixed sized chunking", "Recursive chunking", "Content-aware chunking","Semantic chunking","Agentic chunking"],value="Fixed sized chunking",interactive=True,label="Chunking Algorithm")
        with gr.Group():
            gr.Markdown("Chunking parameters")
            with gr.Row():
                size_input = gr.Number(value=500, label="Size input",interactive=True)
                size_overlap = gr.Number(value=50, label="overlap",interactive=True)
        
        button_chunck = gr.Button("Chunck")
        button_download_store = gr.DownloadButton("Download vector store", value="vector_store.zip")

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
                len_out = gr.Number(value=64, label="Length of output",interactive=True)
        
        button_generate = gr.Button("Generate")
        html_generated_answer_rag_augmenting_context= gr.HTML()
       
        list_saving_otion = ['DOCX', 'PDF', 'HTML']
        saving_option = gr.Dropdown(choices=list_saving_otion,value='PDF',interactive=True,label="Saving option")
        button_save = gr.Button("Save")

if __name__ == "__main__":
    demo.launch(debug=True)




        

