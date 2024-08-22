import streamlit as st
import pdfplumber
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import time
import base64
import re
import roman

# Helper function to read PDF and extract relevant sections
def extract_cv_details(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # Example parsing logic (simplified for demonstration)
    name = re.search(r"([A-Z][a-z]+\s[A-Z][a-z]+)", text).group(1) if re.search(r"([A-Z][a-z]+\s[A-Z][a-z]+)", text) else "Name not found"
    education = re.search(r"EDUCATION(.+?)WORK EXPERIENCE", text, re.S).group(1).strip() if re.search(r"EDUCATION(.+?)WORK EXPERIENCE", text, re.S) else "Education details not found"
    experience = re.search(r"WORK EXPERIENCE(.+?)PROJECTS", text, re.S).group(1).strip() if re.search(r"WORK EXPERIENCE(.+?)PROJECTS", text, re.S) else "Work experience not found"
    projects = re.search(r"PROJECTS(.+?)ADDITIONAL", text, re.S).group(1).strip() if re.search(r"PROJECTS(.+?)ADDITIONAL", text, re.S) else "Projects not found"
    skills = re.search(r"Technical Skills(.+?)Languages", text, re.S).group(1).strip() if re.search(r"Technical Skills(.+?)Languages", text, re.S) else "Skills not found"
    
    return name, education, experience, projects, skills

# Helper function to extract full CV text from PDF
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Helper function to display CV details in a creative manner
def display_cv_details(name, education, experience, projects, skills):
    st.markdown("<h2 style='text-align: center;'>📄 CV Details</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 40px;'>👤 Personal Information</h3>", unsafe_allow_html=True)
    st.header(f"{name}")

    st.markdown("<h3 style='margin-top: 40px;'>🎓 Education</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: justify;'>{education}</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 40px;'>💼 Work Experience</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: justify;'>{experience}</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 40px;'>🔬 Projects</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: justify;'>{projects}</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 40px;'>🛠️ Skills</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: justify;'>{skills}</div>", unsafe_allow_html=True)

# Helper function to scrape job description using Selenium
def extract_job_description_from_url(url):
    # Setup ChromeDriver using WebDriver Manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # Open the webpage
    driver.get(url)
    time.sleep(5)  # Wait for the page to load completely
    
    # Try to find the job description element based on the provided class name
    try:
        job_description = driver.find_element(By.CLASS_NAME, 'styles_JDC__dang-inner-html__h0K4t').text
    except Exception as e:
        job_description = "Job description not found."
    
    # Close the driver
    driver.quit()
    
    return job_description

# Helper function to calculate similarity and top matching keywords
def calculate_similarity(cv_text, job_text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([cv_text, job_text])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    
    # Extract top matching keywords
    feature_names = vectorizer.get_feature_names_out()
    cv_keywords = vectors[0].toarray()[0]
    job_keywords = vectors[1].toarray()[0]
    common_keywords = [(feature_names[i], cv_keywords[i], job_keywords[i])
                       for i in range(len(feature_names)) if cv_keywords[i] > 0 and job_keywords[i] > 0]
    
    # Sort by highest match in both CV and job description
    common_keywords = sorted(common_keywords, key=lambda x: min(x[1], x[2]), reverse=True)
    top_keywords = common_keywords[:top_n]
    
    return cosine_sim, top_keywords, feature_names, job_keywords

# Helper function to suggest missing keywords for CV (with filtering)
def suggest_keywords(cv_text, job_text):
    # Perform TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Use bigrams for more relevance
    vectors = vectorizer.fit_transform([cv_text, job_text])
    
    # Get feature names (keywords) and their importance in the job description
    feature_names = vectorizer.get_feature_names_out()
    cv_keywords = vectors[0].toarray()[0]
    job_keywords = vectors[1].toarray()[0]
    
    # Identify keywords that are important in the job description but missing from the CV
    missing_keywords = [feature_names[i] for i in range(len(feature_names)) if job_keywords[i] > 0 and cv_keywords[i] == 0]
    
    # Filter out non-sensible keywords (numbers, single characters, overly common words)
    filtered_keywords = []
    for keyword in missing_keywords:
        if re.match(r'^[a-zA-Z]+(?: [a-zA-Z]+)*$', keyword) and len(keyword) > 2:  # Filter out numbers and single characters
            filtered_keywords.append(keyword)
    
    # Limit to top 10 missing keywords
    top_missing_keywords = sorted(filtered_keywords, key=lambda x: job_keywords[feature_names.tolist().index(x)], reverse=True)[:10]
    
    return top_missing_keywords

# Helper function to generate and display a bar chart for top keywords
def generate_keyword_bar_chart(top_keywords):
    # Create a DataFrame for the keywords
    df_keywords = pd.DataFrame(top_keywords, columns=['Keyword', 'CV Score', 'Job Score'])
    
    # Sort by job score
    df_keywords = df_keywords.sort_values(by='Job Score', ascending=False)
    
    # Create a bar chart for the job score of the top keywords
    plt.figure(figsize=(10, 6))
    plt.barh(df_keywords['Keyword'], df_keywords['Job Score'], color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Keyword')
    plt.title('Top Keywords from Job Description')
    plt.gca().invert_yaxis()  # Invert y-axis for readability
    st.pyplot(plt)

# Streamlit interface
st.title("CV Match Analyzer")

# Step 1: Upload CV
uploaded_file = st.file_uploader("Upload your CV in PDF format", type="pdf")

# Step 2: Input Job URL(s)
job_urls = st.text_area("Enter one or more job URLs (separated by commas):")

# Step 3: Add a Submit Button
if st.button("Submit"):
    if uploaded_file and job_urls:
        # Extract and display CV details
        name, education, experience, projects, skills = extract_cv_details(uploaded_file)
        display_cv_details(name, education, experience, projects, skills)
        
        # Extract text from CV
        cv_text = extract_text_from_pdf(uploaded_file)
        
        # Handle multiple job URLs
        urls = [url.strip() for url in job_urls.split(",")]
        results = []
        
        for idx, url in enumerate(urls):
            # Extract job description using Selenium
            job_text = extract_job_description_from_url(url)
            
            if job_text != "Job description not found.":
                # Display job description preview
                st.markdown(f"<h2 style='margin-top: 40px;'>Job Analysis {roman.toRoman(idx + 1)}</h2>", unsafe_allow_html=True)
                st.subheader(f"Job Description Preview from {url}:")
                st.write(job_text[:500] + '...')
                
                # Calculate similarity and top matching keywords
                match_rate, top_keywords, feature_names, job_keywords = calculate_similarity(cv_text, job_text)
                results.append({'URL': url, 'Match Rate (%)': f"{match_rate:.2f}"})
                
                # Display result and top keywords
                st.write(f"Your CV matches {match_rate:.2f}% with the job requirements.")
                st.progress(int(match_rate))
                
                st.subheader("Top Matching Keywords:")
                for keyword, cv_score, job_score in top_keywords:
                    st.write(f"- {keyword}: CV score={cv_score:.2f}, Job score={job_score:.2f}")
                
                # Generate and display bar chart for top keywords
                st.subheader("Top Keywords from Job Description")
                generate_keyword_bar_chart(top_keywords)
                
                # Suggest missing keywords for the CV
                st.subheader("🔍 Suggested Keywords to Add to Your CV")
                suggested_keywords = suggest_keywords(cv_text, job_text)
                if suggested_keywords:
                    st.write("Based on the job description, you may want to consider adding the following keywords to your CV:")
                    for keyword in suggested_keywords:
                        st.write(f"- {keyword}")
                else:
                    st.write("Your CV already includes the key terms from the job description.")
            else:
                st.error(f"Job description could not be found for URL: {url}")
        
        # Display all results in a table with no extra blank rows
        st.markdown("<h2 style='margin-top: 40px;'>Summary of Results</h2>", unsafe_allow_html=True)
        result_df = pd.DataFrame(results)
        st.dataframe(result_df)  # Display table with the exact number of rows needed
        
        # Allow user to download results
        csv = result_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="cv_analysis_results.csv">Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Please upload your CV and enter one or more job URLs.")
