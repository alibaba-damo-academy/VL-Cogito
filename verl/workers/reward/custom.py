# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, r1v_length_compute_score, r1v_acc_score


def r1v_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict):

    batch_dict = {}
    batch_dict['rollout_avg_acc']=[]
    batch_dict['rollout_avg_len']=[]
    batch_dict['prompt_avg_acc']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_right']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_false']=[0]*len(batch_dict_reward)
    batch_dict['uid']=[0]*len(batch_dict_reward)
    batch_dict['sample_len']=[]
    batch_dict['sample_acc']=[]

    prompt2sample = prompt_dict
    sample2prompt = {}

    for i in range(len(batch_dict_reward)):
        batch_dict['sample_len'].append(float(batch_valid_response_length[i]))
        batch_dict['sample_acc'].append(batch_dict_reward[i]['acc_reward'])

    roll_avg_acc = sum(batch_dict['sample_acc'])/len(batch_dict['sample_acc'])
    roll_avg_len = float(sum(batch_dict['sample_len'])/len(batch_dict['sample_len']))
    for i in range(len(batch_dict_reward)):
        batch_dict['rollout_avg_acc'].append(roll_avg_acc)
        batch_dict['rollout_avg_len'].append(roll_avg_len)

    for p in prompt_dict.keys():
        one_prompt_acc=[]
        one_prompt_len=[]
        one_prompt_len_right=[]
        one_prompt_len_false=[]
        for s in prompt_dict[p]:
            one_prompt_len.append(batch_dict['sample_len'][s])
            one_prompt_acc.append(batch_dict['sample_acc'][s])
            if batch_dict['sample_acc'][s] == 1:
                one_prompt_len_right.append(batch_dict['sample_len'][s])
            if batch_dict['sample_acc'][s] == 0:
                one_prompt_len_false.append(batch_dict['sample_len'][s])

        for s in prompt_dict[p]:
            batch_dict['uid'][s] = p
            batch_dict['prompt_avg_acc'][s]=sum(one_prompt_acc)/len(one_prompt_acc)
            batch_dict['prompt_avg_len'][s]=float(sum(one_prompt_len)/len(one_prompt_len))
            if len(one_prompt_len_right) != 0:
                batch_dict['prompt_avg_len_right'][s]=float(sum(one_prompt_len_right)/len(one_prompt_len_right))
            else:
                batch_dict['prompt_avg_len_right'][s]=0.0

            if len(one_prompt_len_false) != 0:
                batch_dict['prompt_avg_len_false'][s]=float(sum(one_prompt_len_false)/len(one_prompt_len_false))
            else:
                batch_dict['prompt_avg_len_false'][s]=0.0

    
    batch_final_score=[]
    for i in range(len(batch_dict_reward)):
        batch_final_score.append(batch_dict_reward[i]['all_reward'])

    return batch_final_score, batch_dict


def cos_length(pre, tar, minx=-0.5, maxx=0):
    if pre >= tar*2:
        return minx

    progress = pre / tar
    cosine = math.cos(progress * math.pi)
    # Swap min/max
    min_value = maxx
    max_value = minx

    reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
    return reward


def r1v_dyn_length_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict):

    batch_dict = {}
    batch_dict['rollout_avg_acc']=[]
    batch_dict['rollout_avg_len']=[]
    batch_dict['prompt_avg_acc']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_right']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_false']=[0]*len(batch_dict_reward)
    batch_dict['uid']=[0]*len(batch_dict_reward)
    batch_dict['sample_len']=[]
    batch_dict['sample_acc']=[]

    prompt2sample = prompt_dict
    sample2prompt = {}

    for i in range(len(batch_dict_reward)):
        batch_dict['sample_len'].append(float(batch_valid_response_length[i]))
        batch_dict['sample_acc'].append(batch_dict_reward[i]['acc_reward'])

    roll_avg_acc = sum(batch_dict['sample_acc'])/len(batch_dict['sample_acc'])
    roll_avg_len = float(sum(batch_dict['sample_len'])/len(batch_dict['sample_len']))
    for i in range(len(batch_dict_reward)):
        batch_dict['rollout_avg_acc'].append(roll_avg_acc)
        batch_dict['rollout_avg_len'].append(roll_avg_len)

    for p in prompt_dict.keys():
        one_prompt_acc=[]
        one_prompt_len=[]
        one_prompt_len_right=[]
        one_prompt_len_false=[]
        for s in prompt_dict[p]:
            one_prompt_len.append(batch_dict['sample_len'][s])
            one_prompt_acc.append(batch_dict['sample_acc'][s])
            if batch_dict['sample_acc'][s] == 1:
                one_prompt_len_right.append(batch_dict['sample_len'][s])
            if batch_dict['sample_acc'][s] == 0:
                one_prompt_len_false.append(batch_dict['sample_len'][s])

        for s in prompt_dict[p]:
            batch_dict['uid'][s] = p
            batch_dict['prompt_avg_acc'][s]=sum(one_prompt_acc)/len(one_prompt_acc)
            batch_dict['prompt_avg_len'][s]=float(sum(one_prompt_len)/len(one_prompt_len))
            if len(one_prompt_len_right) != 0:
                batch_dict['prompt_avg_len_right'][s]=float(sum(one_prompt_len_right)/len(one_prompt_len_right))
            else:
                batch_dict['prompt_avg_len_right'][s]=0.0

            if len(one_prompt_len_false) != 0:
                batch_dict['prompt_avg_len_false'][s]=float(sum(one_prompt_len_false)/len(one_prompt_len_false))
            else:
                batch_dict['prompt_avg_len_false'][s]=0.0
    
    batch_final_score=[]
    for i in range(len(batch_dict_reward)):
        one_all_reward = batch_dict_reward[i]['all_reward']
        one_acc_reward = batch_dict_reward[i]['acc_reward']
        one_format_reward = batch_dict_reward[i]['format_reward']
        one_len_reward = batch_dict_reward[i]['len_reward']
        one_rep_reward = batch_dict_reward[i]['rep_reward']

        if batch_dict['prompt_avg_acc'][i] !=0:
            if batch_dict['prompt_avg_acc'][i] <=0.25:
                diff_coef = 1
            else:
                diff_coef = 1
            if one_acc_reward == 1:
                one_final_reward = one_acc_reward*diff_coef+one_format_reward
            else:
                refined_len_reward = cos_length(batch_dict['sample_len'][i], batch_dict['prompt_avg_len_right'][i])
                one_final_reward = one_acc_reward+one_format_reward+refined_len_reward+one_rep_reward
        else:
            one_final_reward = one_acc_reward+one_format_reward+one_len_reward+one_rep_reward
            assert one_final_reward == one_all_reward

        batch_final_score.append(one_final_reward)

    return batch_final_score, batch_dict


def r1v_dyn_length_min_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict):

    batch_dict = {}
    batch_dict['rollout_avg_acc']=[]
    batch_dict['rollout_avg_len']=[]
    batch_dict['prompt_avg_acc']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_right']=[0]*len(batch_dict_reward)
    batch_dict['prompt_avg_len_false']=[0]*len(batch_dict_reward)
    batch_dict['uid']=[0]*len(batch_dict_reward)
    batch_dict['sample_len']=[]
    batch_dict['sample_acc']=[]

    prompt2sample = prompt_dict
    sample2prompt = {}

    for i in range(len(batch_dict_reward)):
        batch_dict['sample_len'].append(float(batch_valid_response_length[i]))
        batch_dict['sample_acc'].append(batch_dict_reward[i]['acc_reward'])

    roll_avg_acc = sum(batch_dict['sample_acc'])/len(batch_dict['sample_acc'])
    roll_avg_len = float(sum(batch_dict['sample_len'])/len(batch_dict['sample_len']))
    for i in range(len(batch_dict_reward)):
        batch_dict['rollout_avg_acc'].append(roll_avg_acc)
        batch_dict['rollout_avg_len'].append(roll_avg_len)

    for p in prompt_dict.keys():
        one_prompt_acc=[]
        one_prompt_len=[]
        one_prompt_len_right=[]
        one_prompt_len_false=[]
        for s in prompt_dict[p]:
            one_prompt_len.append(batch_dict['sample_len'][s])
            one_prompt_acc.append(batch_dict['sample_acc'][s])
            if batch_dict['sample_acc'][s] == 1:
                one_prompt_len_right.append(batch_dict['sample_len'][s])
            if batch_dict['sample_acc'][s] == 0:
                one_prompt_len_false.append(batch_dict['sample_len'][s])

        for s in prompt_dict[p]:
            batch_dict['uid'][s] = p
            batch_dict['prompt_avg_acc'][s]=sum(one_prompt_acc)/len(one_prompt_acc)
            batch_dict['prompt_avg_len'][s]=float(sum(one_prompt_len)/len(one_prompt_len))
            if len(one_prompt_len_right) != 0:
                batch_dict['prompt_avg_len_right'][s]=float(min(one_prompt_len_right))
            else:
                batch_dict['prompt_avg_len_right'][s]=0.0

            if len(one_prompt_len_false) != 0:
                batch_dict['prompt_avg_len_false'][s]=float(sum(one_prompt_len_false)/len(one_prompt_len_false))
            else:
                batch_dict['prompt_avg_len_false'][s]=0.0
    
    batch_final_score=[]
    for i in range(len(batch_dict_reward)):
        one_all_reward = batch_dict_reward[i]['all_reward']
        one_acc_reward = batch_dict_reward[i]['acc_reward']
        one_format_reward = batch_dict_reward[i]['format_reward']
        one_len_reward = batch_dict_reward[i]['len_reward']
        one_rep_reward = batch_dict_reward[i]['rep_reward']

        if batch_dict['prompt_avg_acc'][i] !=0:
            if batch_dict['prompt_avg_acc'][i] <=0.25:
                diff_coef = 2
            else:
                diff_coef = 1
            if one_acc_reward == 1:
                one_final_reward = one_acc_reward*diff_coef+one_format_reward
            else:
                refined_len_reward = cos_length(batch_dict['sample_len'][i], batch_dict['prompt_avg_len_right'][i])
                one_final_reward = one_acc_reward+one_format_reward+refined_len_reward+one_rep_reward
        else:
            one_final_reward = one_acc_reward+one_format_reward+one_len_reward+one_rep_reward
            assert one_final_reward == one_all_reward

        batch_final_score.append(one_final_reward)

    return batch_final_score, batch_dict


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_name = compute_score
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "r1v+cir":
            self.compute_score = r1v_compute_score
        elif compute_score == "r1v+length":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v+length+cir":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v+fixlength":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v+fixlength+cir":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v+minlength":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v+minlength+cir":
            self.compute_score = r1v_length_compute_score
        elif compute_score == "r1v_acc":
            self.compute_score = r1v_acc_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:

        if self.compute_score_name == "r1v_acc":
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print = 0

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["ground_truth"]

                score = self.compute_score(response_str, ground_truth, valid_response_ids)
                reward_tensor[i, valid_response_length - 1] = score['all_reward']

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            return reward_tensor, {}

        if 'batch' not in self.compute_score_name:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print = 0

            #print('!!!!!!!!!!!!!!!!!!!')
            #print('reward input data lenth: ', len(data))
            #print('reward_tensor size: ', reward_tensor.size())
            #print('data keys size: ', data.batch.keys())
            #print(data.non_tensor_batch["uid"])

            prompt_dict = {}
            batch_valid_response_length=[]
            batch_dict_reward=[]

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                
                if data.non_tensor_batch["uid"][i] not in prompt_dict.keys():
                    prompt_dict[data.non_tensor_batch["uid"][i]]=[i]
                else:
                    prompt_dict[data.non_tensor_batch["uid"][i]].append(i)
                

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                batch_valid_response_length.append(valid_response_length)

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["ground_truth"]

                score = self.compute_score(response_str, ground_truth, valid_response_ids, prompt_str = prompt_str)
                batch_dict_reward.append(score)

                #reward_tensor[i, valid_response_length - 1] = score

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            #for k in prompt_dict.keys():
                #print(len(prompt_dict[k]), prompt_dict[k])
            #print(len(prompt_dict.keys()))

            if self.compute_score_name == "r1v" or self.compute_score_name == "r1v_acc" or self.compute_score_name == "r1v+cir":
                batch_final_score, batch_dict=r1v_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict)
            if self.compute_score_name == "r1v+fixlength" or self.compute_score_name == "r1v+fixlength+cir":
                batch_final_score, batch_dict=r1v_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict)
            if self.compute_score_name == "r1v+length":
                batch_final_score, batch_dict=r1v_dyn_length_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict)
            if self.compute_score_name == "r1v+length+cir":
                batch_final_score, batch_dict=r1v_dyn_length_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict)
            if self.compute_score_name == "r1v+minlength" or self.compute_score_name == "r1v+minlength+cir":
                batch_final_score, batch_dict=r1v_dyn_length_min_post_reward(batch_dict_reward, batch_valid_response_length, prompt_dict)
                
            for i in range(len(data.non_tensor_batch["uid"])):
                assert data.non_tensor_batch["uid"][i] == batch_dict["uid"][i]

            for i in range(len(data)):
                reward_tensor[i, batch_valid_response_length[i] - 1] = batch_final_score[i]
            return reward_tensor, batch_dict

        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print = 0

            batch_prompt_ids =[]
            batch_prompt_length =[]
            batch_valid_prompt_length =[]
            batch_valid_prompt_ids =[]
            batch_response_ids =[]
            batch_valid_response_length =[]
            batch_valid_response_ids =[]
            batch_prompt_str =[]
            batch_response_str =[]
            batch_ground_truth =[]

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["ground_truth"]


                #batch 
                batch_prompt_ids.append(prompt_ids)
                batch_prompt_length.append(prompt_length)
                batch_valid_prompt_length.append(valid_prompt_length)
                batch_valid_prompt_ids.append(valid_prompt_ids)
                batch_response_ids.append(response_ids)
                batch_valid_response_length.append(valid_response_length)
                batch_valid_response_ids.append(valid_response_ids)
                batch_prompt_str.append(prompt_str)
                batch_response_str.append(response_str)
                batch_ground_truth.append(ground_truth)

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)

            batch_score = self.compute_score(predict_strs = batch_response_str, ground_truths = batch_ground_truth, prompt_strs = batch_prompt_str, response_length = batch_valid_response_ids)

            for i in range(len(data)):
                score = batch_score[i]
                valid_response_length = batch_valid_response_length[i]
                reward_tensor[i, valid_response_length - 1] = score


            return reward_tensor,{}