import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# Initialize credentials
credentials = Credentials(
    api_key= st.secrets['ibm_api_key'],
    url="https://us-south.ml.cloud.ibm.com"
)

# Set up Streamlit UI
st.title("Personal Poet with IBM Watsonx AI")
st.write("Enter your thoughts and let AI do the Work!.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    'Choose the model:',
    ['GRANITE_20B_MULTILINGUAL', 'GRANITE_13B_CHAT_V2']
)

# User inputs
text = st.text_input('Enter your thoughts:')
length = st.slider('Select the length of the output (in words):', 50, 2000, 1000)
output_type = st.selectbox('Choose the format:', ['Poem', 'Story'])

# Generate button
if st.button('Generate'):
    if text:
        if model_type == 'GRANITE_20B_MULTILINGUAL':
            model_id = ModelTypes.GRANITE_20B_MULTILINGUAL
        else:
            model_id = ModelTypes.GRANITE_13B_CHAT_V2        
        generate_params = {
            GenParams.MAX_NEW_TOKENS: 2000
        }

        model_inference = ModelInference(
            model_id=model_id,
            params=generate_params,
            credentials=credentials,
            project_id=st.secrets['ibm_project_id']
        )

        prompt = f"Generate a fun and enjoyable {length}-word {output_type.lower()} based solely on this person's thoughts: {text}"
        output = model_inference.generate_text(prompt=prompt)

        # Display the generated text
        st.subheader(f"Generated {output_type}:")
        st.write(output)
    else:
        st.warning("Please enter your thoughts before generating the text.")
