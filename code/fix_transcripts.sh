#!/usr/bin/env bash

sed -i '40s/2:50/2:45/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_transcript.txt
sed -i '43s/2:50/2:49/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_transcript.txt

sed -i '64s/2:34/2:38/' stimuli/conv-172/audio/conv-172_run-3_set-2_trial-11_item-12_condition-G_first-A_transcript.txt

# sed -i '5s/M&M's/???/' stimuli/conv-104/audio/conv-104_run-4_set-3_trial-16_item-15_condition-G_first-A_transcript.txt

# TODO conv-134 run 5 trial 18 line 8 has &
# sourcedata/raw_transcripts_from_Revs/CONV_134_run_5_set_3_trial_18_item_18_condition_G_first_B.txt:Yeah.

# grep -norE ' [0-9]+:[0-9]+[\s,]+' sourcedata/raw_transcripts_from_Revs
# sourcedata/raw_transcripts_from_Revs/CONV_168_run_1_set_1_trial_4_item_3_condition_G_first_B.txt:26: 9:30,
# sourcedata/raw_transcripts_from_Revs/CONV_112_run_4_set_3_trial_16_item_16_condition_G_first_A.txt:23: 9:30,

# sourcedata/raw_transcripts_from_Revs/CONV_123_run_5_set_3_trial_18_item_17_condition_G_first_B.txt:2:20,000
# sourcedata/raw_transcripts_from_Revs/CONV_105_run_5_set_3_trial_20_item_19_condition_G_first_B.txt:11:2,000
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_3_set_2_trial_9_item_10_condition_G_first_B.txt:14:1,000
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_18_item_17_condition_G_first_B.txt:2:20,000
# sourcedata/raw_transcripts_from_Revs/CONV_162_run_5_set_3_trial_19_item_19_condition_G_first_A.txt:2:2,000
# sourcedata/raw_transcripts_from_Revs/CONV_162_run_5_set_3_trial_19_item_19_condition_G_first_A.txt:2:2,000

# % grep -norE ' [0-9]+:[0-9]+ ' sourcedata/raw_transcripts_from_Revs
# sourcedata/raw_transcripts_from_Revs/CONV_168_run_1_set_1_trial_4_item_3_condition_G_first_B.txt:20: 6:30 
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:5: 6:00 
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:5: 9:00 
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:5: 9:30 
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:5: 5:30 
# sourcedata/raw_transcripts_from_Revs/CONV_142_run_4_set_3_trial_15_item_16_condition_G_first_A.txt:2: 12:00 
# sourcedata/raw_transcripts_from_Revs/CONV_142_run_4_set_3_trial_15_item_16_condition_G_first_A.txt:2: 4:00 
# sourcedata/raw_transcripts_from_Revs/CONV_175_run_3_set_2_trial_9_item_10_condition_G_first_A.txt:26: 4:00 
# sourcedata/raw_transcripts_from_Revs/CONV_174_run_1_set_1_trial_4_item_4_condition_G_first_B.txt:17: 11:00 
# sourcedata/raw_transcripts_from_Revs/CONV_174_run_1_set_1_trial_4_item_4_condition_G_first_B.txt:17: 11:00 
# sourcedata/raw_transcripts_from_Revs/CONV_174_run_1_set_1_trial_4_item_4_condition_G_first_B.txt:17: 6:00 
# sourcedata/raw_transcripts_from_Revs/CONV_161_run_5_set_3_trial_18_item_18_condition_G_first_B.txt:35: 7:30 
# sourcedata/raw_transcripts_from_Revs/CONV_172_run_4_set_2_trial_14_item_13_condition_G_first_A.txt:14: 5:00 
# sourcedata/raw_transcripts_from_Revs/CONV_175_run_2_set_1_trial_6_item_5_condition_G_first_A.txt:14: 2:00 
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:2: 8:00 
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_5_set_3_trial_19_item_20_condition_G_first_A.txt:8: 8:00 
# sourcedata/raw_transcripts_from_Revs/CONV_131_run_1_set_1_trial_4_item_3_condition_G_first_B.txt:20: 5:00 
# sourcedata/raw_transcripts_from_Revs/CONV_152_run_1_set_1_trial_4_item_4_condition_G_first_A.txt:2: 8:00 
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_1_set_1_trial_1_item_2_condition_G_first_A.txt:29: 1:00 
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_1_set_1_trial_1_item_2_condition_G_first_A.txt:32: 1:00 
# sourcedata/raw_transcripts_from_Revs/CONV_161_run_2_set_1_trial_6_item_5_condition_G_first_A.txt:23: 2:00 

# TODO - remove % from punctuation?
# sourcedata/raw_transcripts_from_Revs/CONV_123_run_3_set_2_trial_11_item_12_condition_G_first_B.txt:20: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_165_run_4_set_3_trial_15_item_16_condition_G_first_B.txt:14: 51%
# sourcedata/raw_transcripts_from_Revs/CONV_165_run_4_set_3_trial_15_item_16_condition_G_first_B.txt:14: 49%
# sourcedata/raw_transcripts_from_Revs/CONV_158_run_1_set_1_trial_4_item_4_condition_G_first_B.txt:20: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_147_run_1_set_1_trial_2_item_1_condition_G_first_A.txt:2: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_173_run_1_set_1_trial_1_item_2_condition_G_first_B.txt:2: 90%
# sourcedata/raw_transcripts_from_Revs/CONV_146_run_3_set_2_trial_9_item_10_condition_G_first_B.txt:14: 5%
# sourcedata/raw_transcripts_from_Revs/CONV_114_run_4_set_3_trial_16_item_16_condition_G_first_A.txt:5: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_114_run_4_set_3_trial_16_item_16_condition_G_first_A.txt:5: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_132_run_5_set_3_trial_18_item_17_condition_G_first_B.txt:8: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_170_run_2_set_1_trial_5_item_5_condition_G_first_A.txt:2: 70%
# sourcedata/raw_transcripts_from_Revs/CONV_114_run_5_set_3_trial_17_item_17_condition_G_first_B.txt:5: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_114_run_5_set_3_trial_17_item_17_condition_G_first_B.txt:5: 90%
# sourcedata/raw_transcripts_from_Revs/CONV_109_run_1_set_1_trial_2_item_2_condition_G_first_A.txt:26: 90%
# sourcedata/raw_transcripts_from_Revs/CONV_169_run_4_set_3_trial_16_item_16_condition_G_first_A.txt:2: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_151_run_3_set_2_trial_10_item_10_condition_G_first_A.txt:14: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_151_run_3_set_2_trial_10_item_10_condition_G_first_A.txt:20: 100%
# sourcedata/raw_transcripts_from_Revs/CONV_147_run_5_set_3_trial_18_item_18_condition_G_first_A.txt:11: 100%