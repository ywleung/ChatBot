Telegram ChatBot: t.me/cornell_ywleung_bot (Server closed due to end of free trial month)

Cornell movie dialogue and OpenSubtitles movie dialogue are used to train this ChatBot. These two datasets aren't uploaded to github because they can be downloaded easily on google.

This ChatBot is based on Selective Model: Embeddings-based ranking, like how Chatterbot works.<br>
I tried to train a Attention-based Seq2Seq model using only Cornell movie dialogue. But it toke too much time to train(about 1 hour for one epoch). After 6 hours of training with GPU GTX960M, the model only responsed "I don't know", "I'm sorry", "yeah"...<br>
Attention-based Seq2Seq model may perform better than selective model if it is trained longer.


The question embedding is split into several files because of the memory limit on Amazon Free tier server.

Training Pipeline:
1) and convert the datasets into question-answer pairs
2) extract useful words by TfidfVectorizer
3) train a supervised embeddings by StarSpace
4) convert questions in train set into question embeddings
5) convert answers(complete sentences) in train set into answer_list.txt 

How the model works:
1) clean the input sentence (remove symbols, stopwords etc.)
2) convert the cleaned sentence into a vector by the StarSpace embeddings
3) find out the closest question in train set
4) return answer which is the response to the closest question in step 3

To-Do List:
1) split train set with different intent/subject(question/non-question, what/when/why/where/how, sport/food/clothes/ar/science ...)
2) make a Cantonese version ChatBot
3) memory network
4) train a Attention-based Seq2Seq model
