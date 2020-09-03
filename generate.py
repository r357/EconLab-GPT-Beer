import gpt_2_simple as gpt2
from datetime import datetime
import os

checkpoint = "run1"

prefixes = ["EconLab beer is ",
            "This beer makes me ",
            "I'd never say this about a beer, but ",
            "EconLab's beer tastes like ",
            "Beer. ",
            "This beer will make you want to ",
            "This beer is ",
            "This beer is just "]




# GENERATE TWEETS

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name=checkpoint)

for prefix in prefixes:
    gpt2.generate(sess,
                    length=250,
                    temperature=0.7,
                    prefix=prefix,
                    nsamples=5,
                    batch_size=5
                    )

