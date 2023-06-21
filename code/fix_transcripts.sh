#!/usr/bin/env bash

# this script is NOT idempotent - do not run it multiple times


# Fix repetition of same time
# ------

# sed -i '6s/126/152/' stimuli/conv-132/transcript/conv-132_run-3_set-2_trial-9_item-10_condition-G_first-A_utterance.csv # The start time of an interval (126.0) cannot occur after its end time (126.0)
# sed -i '7s/129/133/' stimuli/conv-174/transcript/conv-174_run-2_set-1_trial-7_item-7_condition-G_first-B_utterance.csv # The start time of an interval (129.0) cannot occur after its end time (129.0)
# sed -i '9s/168/177/' stimuli/conv-158/transcript/conv-158_run-5_set-3_trial-17_item-17_condition-G_first-A_utterance.csv # The start time of an interval (168.0) cannot occur after its end time (168.0)
# sed -i '6d' stimuli/conv-133/transcript/conv-133_run-1_set-1_trial-3_item-4_condition-G_first-B_utterance.csv # The start time of an interval (123.0) cannot occur after its end time (123.0)
# sed -i '7d' stimuli/conv-128/transcript/conv-128_run-2_set-1_trial-5_item-6_condition-G_first-A_utterance.csv # The start time of an interval (166.0) cannot occur after its end time (166.0)
# sed -i '7s/160/177/' stimuli/conv-137/transcript/conv-137_run-4_set-2_trial-13_item-13_condition-G_first-A_utterance.csv # The start time of an interval (160.0) cannot occur after its end time (160.0)
# sed -i '4s/`//' stimuli/conv-120/transcript/conv-120_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
# sed -i '4s/`//' stimuli/conv-120/transcript/conv-120_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
# sed -i '2s/=/-/' stimuli/conv-143/transcript/conv-143_run-2_set-1_trial-5_item-6_condition-G_first-B_utterance.csv

# redo these:
# sed -i 'xs/2:50/2:45/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_transcript.txt
# sed -i 'xs/2:50/2:49/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_transcript.txt
# sed -i 'xs/2:34/2:38/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-11_item-12_condition-G_first-A_transcript.txt


# look for bracketed items
# grep -ohrE '\[.*\]' sourcedata/transcripts | cut -d' ' -f 1 | sort | uniq -c
# grep -ohrE '\[[^i].*\]' sourcedata/transcripts # not inaudible

# grep -iohrE '\[laugh.*\]' sourcedata/transcripts
# sed -i '2s/\[laughs\]/[laughter]/' stimuli/conv-117/transcript/conv-117_run-5_set-3_trial-12_item-20_condition-G_first-B_utterance.csv
# sed -i '2s/\[Laughs\]/[laughter]/' stimuli/conv-117/transcript/conv-117_run-5_set-3_trial-12_item-20_condition-G_first-B_utterance.csv

# grep -iohrE '\(laugh.*\)' sourcedata/transcripts | cut -d' ' -f1 | sort | uniq -c

# Idiosyncratic fixes
# -------------------

# manually fixed TextGrid after process_transcript:
#   conv-120_run-2_set-1_trial-5_item-5_condition-G_first-A.TextGrid
#   changed line 35 xmax to 171