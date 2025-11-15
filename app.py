import time
import gradio as gr
from generation import generate_rag_response


# ---------------------------------------------------------
# Streaming Response Handler
# ---------------------------------------------------------
def process_query_stream(query, search_type, model_type):
    """
    Stream chunks from generate_rag_response()
    to the Gradio UI smoothly.
    """
    accumulated = ""

    for chunk in generate_rag_response(
        query,
        search_type=search_type,
        top_k=5,
        model_type=model_type,
        stream=True
    ):
        if not chunk:
            continue

        accumulated += chunk

        # smooth UI update
        time.sleep(0.01)
        yield accumulated

    yield accumulated  # final output


# ---------------------------------------------------------
# Non-streaming Response Handler
# ---------------------------------------------------------
def process_query_normal(query, search_type, model_type):
    """
    Return final complete answer.
    """
    return generate_rag_response(
        query,
        search_type=search_type,
        top_k=5,
        model_type=model_type,
        stream=False
    )


# ---------------------------------------------------------
# Wrapper for Gradio submit button
# ---------------------------------------------------------
def on_submit(query, search_type, model_type, stream_enabled):
    """
    Wrapper that chooses streaming or normal mode.
    """
    if not query.strip():
        yield "Please enter a question."
        return

    yield "Retrieving relevant context..."

    if stream_enabled:
        # hand over to streaming generator
        for chunk in process_query_stream(query, search_type, model_type):
            yield chunk
    else:
        # non-streaming mode
        answer = process_query_normal(query, search_type, model_type)
        yield answer


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
with gr.Blocks(title="LocalRAG Q&A System", theme="soft") as demo:
    gr.Markdown("# LocalRAG Q&A System")
    gr.Markdown("Ask questions about the RAG paper and get RAG-powered explanations!")

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about Retrieval-Augmented Generation...",
                lines=4
            )

            with gr.Row():
                search_type = gr.Radio(
                    ["keyword", "semantic", "hybrid"],
                    label="Search Method",
                    value="hybrid",
                )
                model_type = gr.Radio(
                    ["gemini", "ollama"],
                    label="AI Model",
                    value="gemini",
                )

            stream_checkbox = gr.Checkbox(
                label="Stream Response",
                value=True,
                info="Stream the answer in real-time"
            )

            submit_btn = gr.Button("Generate Answer", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Answer",
                lines=20,
                placeholder="Your response will appear here..."
            )

    # Connect button
    submit_btn.click(
        fn=on_submit,
        inputs=[query_input, search_type, model_type, stream_checkbox],
        outputs=output
    )

    # Example prompts
    gr.Examples(
        [
            ["How does RAG work?", "hybrid", "gemini", True],
            ["What are the advantages of RAG vs fine-tuning?", "semantic", "gemini", True],
            ["Explain RAG architecture with diagrams", "hybrid", "ollama", True],
            ["What are common challenges in RAG?", "keyword", "gemini", False],
        ],
        inputs=[query_input, search_type, model_type, stream_checkbox]
    )


# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == "__main__":
    demo.queue().launch()
