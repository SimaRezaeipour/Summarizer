import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns 
from sumy.parsers.plaintext import PlaintextParser
import gensim
#from gensim.summarization import summarize
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
nltk.download('punkt')
#to evaluate summary: Rouge:
from rouge import Rouge
import altair as alt 

def evaluate_summary(summary,reference):
	r = Rouge()
	evaluate_score = r.get_scores(summary,reference)
	evaluate_score_df = pd.DataFrame(evaluate_score[0])
	return evaluate_score_df

def sumy_summarizer(docx,num=2):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,num)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def main():
	st.title("Summerizer App")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menue",menu)
	if choice =="Home":
		st.subheader("Summerization")
		raw_text = st.text_area("Enter your text here:")
		if st.button("Summerize"):
			with st.beta_expander("Original Text"):
				st.write(raw_text)
			c1, c2 = st.beta_columns(2)
			with c1:
				with st.beta_expander("LexRank Summary"):
					my_summary = sumy_summarizer(raw_text)
					document_len = {"Original":len(raw_text),"Summary":len(my_summary)}
					st.write(document_len)
					st.write(my_summary)
					st.info("Rouge Score")
					eval_df = evaluate_summary(my_summary,raw_text)
					st.write(eval_df) #OR: st.dataframe(score)
					eval_df['metrics'] = eval_df.index
					c = alt.Chart(eval_df).mark_bar().encode(x='metrics',y='rouge-1')
					st.altair_chart(c)
			#with c2: 
			#	with st.beta_expander("TextRank Summary"):
			#		my_summary = summarize(raw_text)
			#		st.write(my_summary)


	else:
		st.subheader("About")

if __name__=='__main__':
	main()