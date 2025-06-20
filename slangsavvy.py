import streamlit as st
import google.generativeai as genai
import os
from typing import Optional
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GenerativeAI:
    """
    A class to interface with Google's Generative AI (Gemini) API
    for slang decoding and explanation.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the GenerativeAI class with API key and model.
        
        Args:
            api_key (str): Google Generative AI API key
        """
        self.api_key = api_key
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the pre-trained Gemini model."""
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Try different model names that are currently available
            model_names = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-1.0-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
                'models/gemini-1.0-pro'
            ]
            
            self.model = None
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    test_response = self.model.generate_content("Hello")
                    if test_response:
                        st.success(f"✅ Successfully initialized model: {model_name}")
                        break
                except Exception as model_error:
                    continue
            
            if not self.model:
                # List available models to help debug
                try:
                    available_models = genai.list_models()
                    model_list = [model.name for model in available_models]
                    st.error(f"Could not initialize any model. Available models: {model_list}")
                except:
                    st.error("Could not initialize model or list available models.")
            
            # Configure generation parameters for better responses
            self.generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=1000,
                temperature=0.7,
            )
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            self.model = None
    
    def create_slang_prompt(self, slang_term: str, context: str = "") -> str:
        """
        Create a detailed prompt for slang explanation.
        
        Args:
            slang_term (str): The slang term to decode
            context (str): Optional context where the slang was encountered
            
        Returns:
            str: Formatted prompt for the AI model
        """
        base_prompt = f"""
        You are SlangSavvy, an expert urban slang decoder. Please provide a comprehensive explanation for the slang term: "{slang_term}"

        Please include:
        1. **Definition**: Clear, concise meaning of the term
        2. **Origin**: Where/when this slang originated (if known)
        3. **Usage Examples**: 2-3 realistic examples showing how it's used
        4. **Context**: What platforms/communities commonly use this term
        5. **Variations**: Any alternative spellings or related terms
        6. **Tone**: Whether it's positive, negative, or neutral
        
        """
        
        if context:
            base_prompt += f"\nAdditional context: The user encountered this term in: {context}"
        
        base_prompt += "\nPlease format your response in a clear, engaging way that helps someone understand and use this slang appropriately."
        
        return base_prompt
    
    def get_slang_explanation(self, slang_term: str, context: str = "") -> Optional[str]:
        """
        Get explanation for a slang term from the Gemini model.
        
        Args:
            slang_term (str): The slang term to explain
            context (str): Optional context
            
        Returns:
            Optional[str]: Generated explanation or None if error
        """
        if not self.model:
            return "Error: Model not initialized. Please check your API key."
        
        try:
            # Create the prompt
            prompt = self.create_slang_prompt(slang_term, context)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ai_instance' not in st.session_state:
        st.session_state.ai_instance = None

def display_chat_history():
    """Display the chat history in an attractive format."""
    if st.session_state.chat_history:
        st.subheader("🗨️ Previous Lookups")
        for i, (term, explanation) in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"📱 {term}", expanded=False):
                st.markdown(explanation)

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="SlangSavvy - Urban Slang Decoder",
        page_icon="🔤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🔤 SlangSavvy")
    st.markdown("### *Your AI-Powered Urban Slang Decoder*")
    st.markdown("Navigate the world of internet slang with confidence! 🌐✨")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Try to get API key from environment first
        env_api_key = os.getenv("GEMINI_API_KEY")
        
        if env_api_key:
            st.success("✅ API Key loaded from .env file")
            api_key = env_api_key
            st.info(f"Using API key: {api_key[:8]}{'*' * (len(api_key) - 8)}")
        else:
            st.warning("⚠️ No API key found in .env file")
            # API Key input as fallback
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Enter your Google Generative AI API key (or add GEMINI_API_KEY to .env file)"
            )
        
        if api_key:
            if st.session_state.ai_instance is None or st.session_state.get('current_api_key') != api_key:
                st.session_state.current_api_key = api_key
                with st.spinner("Initializing AI model..."):
                    st.session_state.ai_instance = GenerativeAI(api_key)
                    time.sleep(1)  # Give it a moment to initialize
        
        # Add a button to list available models for debugging
        if st.button("🔍 List Available Models"):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    models = genai.list_models()
                    st.write("Available models:")
                    for model in models:
                        st.write(f"- {model.name}")
                except Exception as e:
                    st.error(f"Error listing models: {e}")
        
        st.markdown("---")
        st.markdown("### 🔧 Setup Instructions:")
        st.markdown("**Option 1: Using .env file (Recommended)**")
        st.code("""
# Create a .env file in your project root:
GEMINI_API_KEY=your_api_key_here
        """)
        st.markdown("**Option 2: Manual input**")
        st.markdown("• Enter your API key in the field above")
        
        st.markdown("---")
        st.markdown("### 🔧 Troubleshooting:")
        st.markdown("• Make sure your API key is valid")
        st.markdown("• Check if your API key has Gemini access enabled")
        st.markdown("• Try clicking 'List Available Models' to see what's available")
        
        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown("1. Add your API key to .env file OR enter it manually")
        st.markdown("2. Type any slang term you want to understand")
        st.markdown("3. Add context if you have it (optional)")
        st.markdown("4. Click 'Decode Slang' to get explanation")
        
        st.markdown("---")
        st.markdown("### 💡 Example terms:")
        st.markdown("• slay\n• bet\n• cap\n• periodt\n• stan\n• vibe check")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Check if API key is provided
        if not api_key:
            st.warning("👈 Please enter your Google Gemini API key in the sidebar to get started!")
            st.info("Don't have an API key? Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)")
            return
        
        if not st.session_state.ai_instance or not st.session_state.ai_instance.model:
            st.error("Please check your API key and try again.")
            return
        
        # Input form
        with st.form("slang_form"):
            st.subheader("🔤 Enter Slang Term")
            
            slang_input = st.text_input(
                "Slang Term",
                placeholder="e.g., slay, bet, cap, periodt...",
                help="Enter the slang term you want to understand"
            )
            
            context_input = st.text_area(
                "Context (Optional)",
                placeholder="Where did you see this? e.g., 'on TikTok', 'in a gaming chat', 'on Twitter'...",
                help="Providing context helps get more accurate explanations",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Decode Slang", use_container_width=True)
        
        # Process the form submission
        if submitted and slang_input.strip():
            with st.spinner("🔍 Decoding slang..."):
                explanation = st.session_state.ai_instance.get_slang_explanation(
                    slang_input.strip(), 
                    context_input.strip()
                )
                
                if explanation:
                    # Display the result
                    st.subheader(f"📱 Explanation for: *{slang_input}*")
                    st.markdown(explanation)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((slang_input, explanation))
                    
                    # Success message
                    st.success("✅ Slang decoded successfully!")
                else:
                    st.error("❌ Failed to decode slang. Please try again.")
        
        elif submitted and not slang_input.strip():
            st.warning("⚠️ Please enter a slang term to decode!")
    
    with col2:
        # Fun facts or trending slang
        st.subheader("📈 Did You Know?")
        st.info("🔥 Internet slang evolves rapidly! New terms appear daily on platforms like TikTok, Twitter, and Discord.")
        
        st.subheader("🎯 Perfect for:")
        st.markdown("""
        • 📱 Social media browsing
        • 🎮 Gaming communities  
        • 💬 Online forums
        • 📺 Understanding memes
        • 🎓 Staying current with trends
        """)
    
    # Display chat history at the bottom
    if st.session_state.chat_history:
        st.markdown("---")
        display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔤 SlangSavvy - Powered by Google Gemini AI | Stay connected with digital culture!"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    