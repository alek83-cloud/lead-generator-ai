import streamlit as st
import os

def render_sidebar():
    """Render sidebar with provider/model selection and an API test button."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # ---------- Provider ----------
        provider = st.selectbox(
            "AI Provider",
            ["OpenAI", "Gemini"],
            help="Choose the LLM provider"
        )
        provider_lower = provider.lower()

        # ---------- Model ----------
        if provider_lower == "openai":
            model_choice = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini"],
                help="OpenAI GPT-4 family models"
            )
        else:
            model_choice = st.selectbox(
                "Model",
                ["gemini-1.5-pro", "gemini-1.5-flash"],
                help="Google Gemini 1.5 family models"
            )

        # LiteLLM needs provider/model format
        model = f"{provider_lower}/{model_choice}"

        # ---------- API Keys ----------
        with st.expander("🔑 API Keys", expanded=True):
            st.info("Keys live only in memory and are cleared when you close the browser.")
            openai_api_key = ""
            gemini_api_key = ""
            if provider_lower == "openai":
                openai_api_key = st.text_input(
                    "OpenAI API Key", type="password",
                    placeholder="Enter your OpenAI key"
                )
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                gemini_api_key = st.text_input(
                    "Gemini API Key", type="password",
                    placeholder="Enter your Gemini key"
                )
                if gemini_api_key:
                    os.environ["GEMINI_API_KEY"] = gemini_api_key

            serp_api_key = st.text_input(
                "Serper API Key", type="password",
                placeholder="Enter your Serper API key",
                help="For optional web search enrichment"
            )
            if serp_api_key:
                os.environ["SERPER_API_KEY"] = serp_api_key

        # ---------- Quick API connectivity test ----------
        st.divider()
        st.subheader("🔌 Test API Connection")

        if st.button("Run quick ping test"):
            try:
                if provider_lower == "openai":
                    import openai
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    if not openai.api_key:
                        st.error("OpenAI key missing.")
                    else:
                        resp = openai.ChatCompletion.create(
                            model=model_choice,
                            messages=[{"role": "user", "content": "Return a single word: ping"}],
                            max_tokens=5
                        )
                        st.success(f"✅ OpenAI OK: {resp.choices[0].message['content']}")
                else:
                    from google import generativeai as genai
                    key = os.getenv("GEMINI_API_KEY")
                    if not key:
                        st.error("Gemini key missing.")
                    else:
                        genai.configure(api_key=key)
                        gmodel = genai.GenerativeModel(model_choice)
                        resp = gmodel.generate_content("Return a single word: ping")
                        st.success(f"✅ Gemini OK: {resp.text}")
            except Exception as e:
                st.error(f"❌ API test failed: {e}")

        # ---------- About ----------
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
                This app can run on **OpenAI or Gemini**.  
                Use the button above to quickly check connectivity
                before launching a full lead-generation run.
            """)

    return {
        "provider": provider,
        "model": model,  # always prefixed for LiteLLM/CrewAI
        "has_openai_key": bool(openai_api_key),
        "has_gemini_key": bool(gemini_api_key),
        "has_serp_key": bool(serp_api_key)
    }
