#!/usr/bin/env bash

# NOTE: this script is NOT idempotent (do not run it multiple times)

set -e
trap 'echo Failed on line: $LINENO at command: $BASH_COMMAND && exit $?' ERR

# ---------------
# switch speakers
# ---------------
# mainly fixes issues found after `copy_transcripts.py` and some typos

sed -i '3s/103/3/' stimuli/conv-103/transcript/conv-103_run-3_set-1_trial-7_item-12_condition-G_first-B_utterance.csv
sed -i '5s/103/3/' stimuli/conv-103/transcript/conv-103_run-3_set-1_trial-7_item-12_condition-G_first-B_utterance.csv

sed -i '4s/5/105/' stimuli/conv-105/transcript/conv-105_run-1_set-1_trial-1_item-1_condition-G_first-B_utterance.csv
sed -i '6s/5/105/' stimuli/conv-105/transcript/conv-105_run-1_set-1_trial-1_item-1_condition-G_first-B_utterance.csv
sed -i '8s/5/105/' stimuli/conv-105/transcript/conv-105_run-1_set-1_trial-1_item-1_condition-G_first-B_utterance.csv

sed -i '10s/106/6/' stimuli/conv-106/transcript/conv-106_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv

sed -i '5s/13/113/' stimuli/conv-113/transcript/conv-113_run-2_set-1_trial-8_item-7_condition-G_first-B_utterance.csv

sed -i '4s/127/27/' stimuli/conv-127/transcript/conv-127_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '6s/127/27/' stimuli/conv-127/transcript/conv-127_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv

sed -i '7d' stimuli/conv-128/transcript/conv-128_run-2_set-1_trial-5_item-6_condition-G_first-A_utterance.csv
sed -i '7s/28/128/' stimuli/conv-128/transcript/conv-128_run-5_set-3_trial-17_item-17_condition-G_first-B_utterance.csv

sed -i '4s/130/30/' stimuli/conv-130/transcript/conv-130_run-3_set-2_trial-9_item-9_condition-G_first-A_utterance.csv
sed -i '5s/130/30/' stimuli/conv-130/transcript/conv-130_run-4_set-2_trial-14_item-14_condition-G_first-B_utterance.csv

sed -i '4s/134/34/' stimuli/conv-134/transcript/conv-134_run-1_set-1_trial-2_item-1_condition-G_first-A_utterance.csv
sed -i '6s/134/34/' stimuli/conv-134/transcript/conv-134_run-1_set-1_trial-2_item-1_condition-G_first-A_utterance.csv
sed -i '8s/134/34/' stimuli/conv-134/transcript/conv-134_run-1_set-1_trial-2_item-1_condition-G_first-A_utterance.csv

sed -i '3s/143/43/' stimuli/conv-143/transcript/conv-143_run-4_set-3_trial-8_item-15_condition-G_first-B_utterance.csv

sed -i '4s/47/147/' stimuli/conv-147/transcript/conv-147_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv
sed -i '5d' stimuli/conv-147/transcript/conv-147_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv

sed -i '4s/48/148/' stimuli/conv-148/transcript/conv-148_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '6s/48/148/' stimuli/conv-148/transcript/conv-148_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '8s/48/148/' stimuli/conv-148/transcript/conv-148_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '6s/vacuum/vector/' stimuli/conv-148/transcript/conv-148_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '5s/48/148/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv
sed -i '6s/48/148/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv
sed -i '8s/48/148/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv
sed -i '9s/48/148/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv
sed -i '11s/48/148/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv
sed -i '5s/there/either/' stimuli/conv-148/transcript/conv-148_run-2_set-1_trial-6_item-5_condition-G_first-A_utterance.csv

sed -i '3s/53/153/' stimuli/conv-153/transcript/conv-153_run-5_set-3_trial-18_item-17_condition-G_first-A_utterance.csv
sed -i '4s/53/153/' stimuli/conv-153/transcript/conv-153_run-5_set-3_trial-18_item-17_condition-G_first-A_utterance.csv
sed -i '6s/53/153/' stimuli/conv-153/transcript/conv-153_run-5_set-3_trial-18_item-17_condition-G_first-A_utterance.csv
sed -i '8s/53/153/' stimuli/conv-153/transcript/conv-153_run-5_set-3_trial-18_item-17_condition-G_first-A_utterance.csv

sed -i '4s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '6s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '8s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '10s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '12s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '14s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '16s/172,1/72,1/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '18s/172/72/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '5s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '7s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '9s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '11s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '13s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '15s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '17s/72/172/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '31d' stimuli/conv-172/transcript/conv-172_run-5_set-3_trial-20_item-20_condition-G_first-A_utterance.csv

sed -i '5s/73/173/' stimuli/conv-173/transcript/conv-173_run-2_set-1_trial-8_item-7_condition-G_first-B_utterance.csv

# --------------------
# fix incorrect timing
# --------------------
# mainly fixes issues found in `process_transcripts.py`
# typically these are incorrect starting times of utterances

sed -i '6s/100/71/' stimuli/conv-103/transcript/conv-103_run-1_set-1_trial-2_item-2_condition-G_first-B_utterance.csv
sed -i '7s/100/72/' stimuli/conv-103/transcript/conv-103_run-1_set-1_trial-2_item-2_condition-G_first-B_utterance.csv
sed -i '8s/100/87/' stimuli/conv-103/transcript/conv-103_run-1_set-1_trial-2_item-2_condition-G_first-B_utterance.csv
# why did this whole conv need retiming?
sed -i '4s/36/58/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '5s/36/83/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '8s/113/124/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '9s/113/134/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '10s/118/144/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv
sed -i '11s/118/167/' stimuli/conv-103/transcript/conv-103_run-2_set-1_trial-4_item-7_condition-G_first-A_utterance.csv

sed -i '16s/106/116/' stimuli/conv-109/transcript/conv-109_run-4_set-3_trial-15_item-15_condition-G_first-B_utterance.csv
sed -i '18s/121/125/' stimuli/conv-109/transcript/conv-109_run-4_set-3_trial-15_item-15_condition-G_first-B_utterance.csv
sed -i '23s/155/160/' stimuli/conv-109/transcript/conv-109_run-4_set-3_trial-15_item-15_condition-G_first-B_utterance.csv

sed -i '7d' stimuli/conv-112/transcript/conv-112_run-4_set-3_trial-16_item-16_condition-G_first-A_utterance.csv

sed -i '7s/77/101/' stimuli/conv-113/transcript/conv-113_run-5_set-3_trial-18_item-17_condition-G_first-B_utterance.csv

sed -i '8s/173/177/' stimuli/conv-130/transcript/conv-130_run-1_set-1_trial-4_item-3_condition-G_first-B_utterance.csv

sed -i '6s/126/152/' stimuli/conv-132/transcript/conv-132_run-3_set-2_trial-9_item-10_condition-G_first-A_utterance.csv # The start time of an interval (126.0) cannot occur after its end time (126.0)

sed -i '6d' stimuli/conv-133/transcript/conv-133_run-1_set-1_trial-3_item-4_condition-G_first-B_utterance.csv # The start time of an interval (123.0) cannot occur after its end time (123.0)

sed -i '6s/121/178/' stimuli/conv-134/transcript/conv-134_run-4_set-3_trial-15_item-16_condition-G_first-B_utterance.csv

sed -i '7s/160/177/' stimuli/conv-137/transcript/conv-137_run-4_set-2_trial-13_item-13_condition-G_first-A_utterance.csv # The start time of an interval (160.0) cannot occur after its end time (160.0)

sed -i '11s/162/157/' stimuli/conv-146/transcript/conv-146_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
sed -i '12s/162/164/' stimuli/conv-146/transcript/conv-146_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
sed -i '15s/156/177/' stimuli/conv-146/transcript/conv-146_run-3_set-2_trial-11_item-11_condition-G_first-A_utterance.csv
sed -i '3s/3/17/' stimuli/conv-146/transcript/conv-146_run-5_set-3_trial-19_item-20_condition-G_first-A_utterance.csv
sed -i '4s/28/20/' stimuli/conv-146/transcript/conv-146_run-5_set-3_trial-19_item-20_condition-G_first-A_utterance.csv

sed -i '5d' stimuli/conv-152/transcript/conv-152_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv

sed -i '7s/47/46/' stimuli/conv-154/transcript/conv-154_run-1_set-1_trial-2_item-2_condition-G_first-B_utterance.csv

sed -i '15s/161/163/' stimuli/conv-155/transcript/conv-155_run-4_set-2_trial-14_item-13_condition-G_first-A_utterance.csv

sed -i '9s/168/177/' stimuli/conv-158/transcript/conv-158_run-5_set-3_trial-17_item-17_condition-G_first-A_utterance.csv # The start time of an interval (168.0) cannot occur after its end time (168.0)

sed -i '9d' stimuli/conv-159/transcript/conv-159_run-2_set-1_trial-8_item-7_condition-G_first-B_utterance.csv

sed -i '13s/147/165/' stimuli/conv-160/transcript/conv-160_run-2_set-1_trial-5_item-5_condition-G_first-A_utterance.csv
sed -i '6d' stimuli/conv-160/transcript/conv-160_run-2_set-1_trial-5_item-5_condition-G_first-A_utterance.csv

sed -i '7s/125/137/' stimuli/conv-161/transcript/conv-161_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '7s/125/137/' stimuli/conv-161/transcript/conv-161_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv
sed -i '14s/171/174/' stimuli/conv-161/transcript/conv-161_run-1_set-1_trial-2_item-2_condition-G_first-A_utterance.csv

sed -i '8d' stimuli/conv-164/transcript/conv-164_run-4_set-2_trial-13_item-14_condition-G_first-A_utterance.csv
# Long silence at beginning, and halucinated transcription?
sed -i '3s/27/10/' stimuli/conv-164/transcript/conv-164_run-4_set-3_trial-15_item-16_condition-G_first-B_utterance.csv
sed -i '2d' stimuli/conv-164/transcript/conv-164_run-4_set-3_trial-15_item-16_condition-G_first-B_utterance.csv

sed -i '12s/162/169/' stimuli/conv-165/transcript/conv-165_run-1_set-1_trial-1_item-1_condition-G_first-B_utterance.csv
sed -i '14s/159/154/' stimuli/conv-165/transcript/conv-165_run-1_set-1_trial-3_item-4_condition-G_first-B_utterance.csv
sed -i '12s/135/138/' stimuli/conv-165/transcript/conv-165_run-1_set-1_trial-3_item-4_condition-G_first-B_utterance.csv
sed -i '10s/167/174/' stimuli/conv-165/transcript/conv-165_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv

sed -i '26s/171/169/' stimuli/conv-166/transcript/conv-166_run-1_set-1_trial-3_item-3_condition-G_first-B_utterance.csv
sed -i '26s/\[inaudible\]/Oh geez. What if I start thinking about [inaudible]?/' stimuli/conv-166/transcript/conv-166_run-1_set-1_trial-3_item-3_condition-G_first-B_utterance.csv
sed -i '27s/171/173/' stimuli/conv-166/transcript/conv-166_run-1_set-1_trial-3_item-3_condition-G_first-B_utterance.csv
sed -i '28d' stimuli/conv-166/transcript/conv-166_run-1_set-1_trial-3_item-3_condition-G_first-B_utterance.csv
sed -i '21s/146/147/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-6_item-6_condition-G_first-A_utterance.csv
sed -i '22s/146/151/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-6_item-6_condition-G_first-A_utterance.csv
sed -i '24s/161/158/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-6_item-6_condition-G_first-A_utterance.csv
sed -i '25s/161/159/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-6_item-6_condition-G_first-A_utterance.csv
sed -i '6s/36/43/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv
sed -i '12s/82/95/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv
sed -i '13s/94/96/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv
sed -i '23s/169/172/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv
sed -i '24s/169/174/' stimuli/conv-166/transcript/conv-166_run-2_set-1_trial-8_item-8_condition-G_first-A_utterance.csv
# NOTE singing in the following parts _can_ be transcribed, not [inaudible]
sed -i '21s/101/106/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-10_item-9_condition-G_first-A_utterance.csv
sed -i '22s/106/109/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-10_item-9_condition-G_first-A_utterance.csv
sed -i '23s/106/111/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-10_item-9_condition-G_first-A_utterance.csv
sed -i '34s/174/169/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-10_item-9_condition-G_first-A_utterance.csv
sed -i '35s/175/173/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-10_item-9_condition-G_first-A_utterance.csv
sed -i '7s/17/18/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '16s/58/62/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '17s/58/68/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '26s/124/122/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '27s/126/123/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '28s/126/124/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '29s/129/125/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '30s/130/127/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '32s/130/131/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '33s/145/135/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '34s/145/143/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '35s/145/144/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '42s/170/173/' stimuli/conv-166/transcript/conv-166_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '14s/96/115/' stimuli/conv-166/transcript/conv-166_run-4_set-2_trial-14_item-14_condition-G_first-B_utterance.csv
sed -i '15s/96/118/' stimuli/conv-166/transcript/conv-166_run-4_set-2_trial-14_item-14_condition-G_first-B_utterance.csv
sed -i '16s/113/120/' stimuli/conv-166/transcript/conv-166_run-4_set-2_trial-14_item-14_condition-G_first-B_utterance.csv
sed -i '6s/53/56/' stimuli/conv-166/transcript/conv-166_run-4_set-3_trial-16_item-15_condition-G_first-B_utterance.csv
sed -i '7s/53/59/' stimuli/conv-166/transcript/conv-166_run-4_set-3_trial-16_item-15_condition-G_first-B_utterance.csv
sed -i '14s/80/85/' stimuli/conv-166/transcript/conv-166_run-4_set-3_trial-16_item-15_condition-G_first-B_utterance.csv
sed -i '17s/91/102/' stimuli/conv-166/transcript/conv-166_run-4_set-3_trial-16_item-15_condition-G_first-B_utterance.csv
sed -i '7s/59/64/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-17_item-18_condition-G_first-B_utterance.csv
sed -i '18s/124/126/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-17_item-18_condition-G_first-B_utterance.csv
sed -i '22s/144/150/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-17_item-18_condition-G_first-B_utterance.csv
sed -i '23s/150/152/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-17_item-18_condition-G_first-B_utterance.csv
sed -i '7s/73/67/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv
sed -i '8s/73/70/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv
sed -i '32s/160/161/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv
sed -i '33s/160/164/' stimuli/conv-166/transcript/conv-166_run-5_set-3_trial-20_item-19_condition-G_first-B_utterance.csv

sed -i '7s/125/106/' stimuli/conv-170/transcript/conv-170_run-2_set-1_trial-5_item-5_condition-G_first-A_utterance.csv
sed -i '8s/125/120/' stimuli/conv-170/transcript/conv-170_run-2_set-1_trial-5_item-5_condition-G_first-A_utterance.csv
sed -i '6d' stimuli/conv-170/transcript/conv-170_run-2_set-1_trial-5_item-5_condition-G_first-A_utterance.csv
sed -i '5d' stimuli/conv-170/transcript/conv-170_run-3_set-2_trial-12_item-12_condition-G_first-A_utterance.csv
sed -i '3s/4/10/' stimuli/conv-170/transcript/conv-170_run-3_set-2_trial-9_item-9_condition-G_first-A_utterance.csv
sed -i '6s/164/174/' stimuli/conv-170/transcript/conv-170_run-5_set-3_trial-19_item-20_condition-G_first-A_utterance.csv

sed -i '10s/152/160/' stimuli/conv-172/transcript/conv-172_run-2_set-1_trial-5_item-6_condition-G_first-B_utterance.csv
sed -i '15s/170/165/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
sed -i '23s/154/158/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-11_item-12_condition-G_first-A_utterance.csv
sed -i '23s/154/158/' stimuli/conv-172/transcript/conv-172_run-3_set-2_trial-11_item-12_condition-G_first-A_utterance.csv
sed -i '8s/35/36/' stimuli/conv-172/transcript/conv-172_run-4_set-3_trial-15_item-16_condition-G_first-B_utterance.csv
sed -i '20s/168/175/' stimuli/conv-172/transcript/conv-172_run-4_set-3_trial-15_item-16_condition-G_first-B_utterance.csv

sed -i '7s/129/133/' stimuli/conv-174/transcript/conv-174_run-2_set-1_trial-7_item-7_condition-G_first-B_utterance.csv # The start time of an interval (129.0) cannot occur after its end time (129.0)

sed -i '14s/177/171/' stimuli/conv-175/transcript/conv-175_run-4_set-2_trial-14_item-13_condition-G_first-A_utterance.csv
sed -i '12s/168/173/' stimuli/conv-175/transcript/conv-175_run-4_set-3_trial-15_item-15_condition-G_first-B_utterance.csv
sed -i '13s/168/177/' stimuli/conv-175/transcript/conv-175_run-4_set-3_trial-15_item-15_condition-G_first-B_utterance.csv
sed -i '10s/141/149/' stimuli/conv-175/transcript/conv-175_run-5_set-3_trial-18_item-18_condition-G_first-B_utterance.csv


# --------------------
# misc fixes
# --------------------
# grep -iohrE '\[laugh.*\]' sourcedata/transcripts
# look for bracketed items
# grep -ohrE '\[.*\]' sourcedata/transcripts | cut -d' ' -f 1 | sort | uniq -c
# grep -ohrE '\[[^i].*\]' sourcedata/transcripts # not inaudible
# grep -iohrE '\(laugh.*\)' sourcedata/transcripts | cut -d' ' -f1 | sort | uniq -c

# manually fixed TextGrid after process_transcript:
#   conv-120_run-2_set-1_trial-5_item-5_condition-G_first-A.TextGrid
#   changed line 35 xmax to 171

sed -i '2s/\[laughs\]/[laughter]/' stimuli/conv-117/transcript/conv-117_run-5_set-3_trial-12_item-20_condition-G_first-B_utterance.csv
sed -i '2s/\[Laughs\]/[laughter]/' stimuli/conv-117/transcript/conv-117_run-5_set-3_trial-12_item-20_condition-G_first-B_utterance.csv

sed -i '2s/=/-/' stimuli/conv-143/transcript/conv-143_run-2_set-1_trial-5_item-6_condition-G_first-B_utterance.csv
sed -i '4s/`//' stimuli/conv-120/transcript/conv-120_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
sed -i '4s/`//' stimuli/conv-120/transcript/conv-120_run-1_set-1_trial-3_item-4_condition-G_first-A_utterance.csv
