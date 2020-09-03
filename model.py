import gpt_2_simple as gpt2
import os

# GPT Model, tweets, session ('fresh' or 'latest' for restore)
model_name = "124M"
tweets = "kanyewest_tweets.csv"
session = "fresh"
run_name = 'run1'



# Check model
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)


# Paths and naming operations
tweets_path = os.path.join("Tweets", tweets)
out_name = tweets.split("_")[0] + "_generated"


# Start TF Session
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
				dataset = tweets_path,
				model_name = model_name,
				steps = 1000,
				restore_from = session,
				run_name = run_name,
				print_every = 10,
				sample_every = 200,
				save_every = 500
				)
