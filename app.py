import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
import PyPDF2
import pytesseract
from PIL import Image
import io

# --- AGENT CLASSES ---

class FileExtractorAgent:
    def process_file(self, uploaded_file):
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_type in ['txt']:
                return self._process_text(uploaded_file)
            elif file_type in ['xlsx', 'csv']:
                return self._process_tabular(uploaded_file)
            elif file_type == 'pdf':
                return self._process_pdf(uploaded_file)
            elif file_type in ['jpg', 'jpeg', 'png']:
                return self._process_image(uploaded_file)
            else:
                return "Unsupported file format"
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def _process_text(self, file):
        return file.getvalue().decode("utf-8", errors="ignore")

    def _process_tabular(self, file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        return pd.read_excel(file)

    def _process_pdf(self, file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text

    def _process_image(self, file):
        image = Image.open(file)
        return pytesseract.image_to_string(image)

class QuestionAnsweringAgent:
    def __init__(self):
        # Use Streamlit secrets for API key
        self.client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

    def answer_question(self, content, question):
        prompt = f"""Document content:
{content}

Question: {question}
Answer:"""
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

class VisualizationAgent:
    def generate_visualization(self, data, instruction):
        fig, ax = plt.subplots()
        if isinstance(data, pd.DataFrame):
            # Basic heuristics for visualization
            if "histogram" in instruction.lower():
                num_cols = data.select_dtypes(include='number').columns
                if len(num_cols) > 0:
                    data[num_cols[0]].plot.hist(ax=ax)
                    ax.set_title(f"Histogram of {num_cols[0]}")
                else:
                    ax.text(0.5, 0.5, "No numeric columns for histogram", ha='center', va='center')
            elif "scatter" in instruction.lower() or "vs" in instruction.lower():
                cols = data.select_dtypes(include='number').columns
                if len(cols) >= 2:
                    sns.scatterplot(x=cols[0], y=cols[1], data=data, ax=ax)
                    ax.set_title(f"Scatter plot: {cols[0]} vs {cols[1]}")
                else:
                    ax.text(0.5, 0.5, "Not enough numeric columns for scatter plot", ha='center', va='center')
            elif "bar" in instruction.lower() or "plot" in instruction.lower() or "chart" in instruction.lower():
                cols = data.columns
                if len(cols) >= 2:
                    data.groupby(cols[0])[cols[1]].sum().plot.bar(ax=ax)
                    ax.set_title(f"Bar chart: {cols[1]} by {cols[0]}")
                else:
                    ax.text(0.5, 0.5, "Not enough columns for bar chart", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Visualization type not recognized.", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No DataFrame available for visualization.", ha='center', va='center')
        plt.tight_layout()
        return fig

# --- ORCHESTRATOR ---

class Orchestrator:
    def __init__(self):
        self.file_agent = FileExtractorAgent()
        self.qa_agent = QuestionAnsweringAgent()
        self.viz_agent = VisualizationAgent()
        self.content = None

    def process_file(self, uploaded_file):
        self.content = self.file_agent.process_file(uploaded_file)
        return self.content

    def handle_query(self, query):
        if self.content is None:
            return "Please upload and process a file first."
        # Visualization keywords
        viz_keywords = ['plot', 'chart', 'graph', 'histogram', 'bar', 'scatter', 'visualize', 'show']
        if any(word in query.lower() for word in viz_keywords):
            if isinstance(self.content, pd.DataFrame):
                fig = self.viz_agent.generate_visualization(self.content, query)
                return fig
            else:
                return "Visualization is only supported for tabular data (CSV/XLSX)."
        else:
            # For text or DataFrame, convert DataFrame to string for LLM
            content_str = self.content if isinstance(self.content, str) else self.content.head(100).to_csv(index=False)
            answer = self.qa_agent.answer_question(content_str, query)
            return answer

# --- STREAMLIT UI ---

st.set_page_config(page_title="Multi-Agent Data Analyst", layout="wide")
st.title("📊 Multi-Agent Data Analyst Assistant")

st.markdown("""
Upload a document (.doc, .txt, .xlsx, .csv, .pdf, or image file), ask questions about its contents, or request visualizations.
""")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv", "xlsx", "pdf", "jpg", "jpeg", "png"])

if uploaded_file and not st.session_state.file_processed:
    with st.spinner("Processing file..."):
        content = st.session_state.orchestrator.process_file(uploaded_file)
        st.session_state.file_processed = True
    st.success("File processed successfully!")
    if isinstance(content, pd.DataFrame):
        st.dataframe(content.head())
    else:
        st.text_area("Extracted Content Preview", content[:1000] if isinstance(content, str) else str(content)[:1000], height=200)

if st.session_state.file_processed:
    query = st.text_input("Ask a question or request a visualization (e.g., 'Show a histogram of sales'):")
    if query:
        with st.spinner("Processing your query..."):
            result = st.session_state.orchestrator.handle_query(query)
        if isinstance(result, plt.Figure):
            st.pyplot(result)
        else:
            st.write(result)

    if st.button("Reset"):
        st.session_state.file_processed = False
        st.session_state.orchestrator = Orchestrator()
        st.experimental_rerun()

st.markdown("---")
st.caption("Powered by Together AI, Llama-4-Maverick-17B, and Streamlit. [Source on GitHub](https://github.com/yourusername/your-repo)")
