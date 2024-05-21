

import time
import argparse
from model.ea_model import EaModel
import torch
from fastchat.model import get_conversation_template
from datasets import load_dataset
from tqdm import tqdm


def truncate_list(lst, num):
    if num not in lst:
        return lst

    first_index = lst.index(num)
    return lst[:first_index + 1]


def warmup(args, model):
    conv = get_conversation_template(args.model_type)
    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p

    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"

    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    if args.model_type == "llama-2-chat":
        prompt += " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    for output_ids in model.ea_generate(input_ids):
        ol = output_ids.shape[1]


def test(args, model, input_text, temperature, top_p, use_eaInfer, use_sysm_prompt, case_key=""):
    assert args.model_type == "llama-2-chat" or "mixtral"
    conv = get_conversation_template(args.model_type)
    if use_sysm_prompt:
        # conv = get_conversation_template(args.model_type)

        if args.model_type == "llama-2-chat":
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        elif args.model_type == "mixtral":
            conv = get_conversation_template("llama-2-chat")
            conv.system_message = ""
            conv.sep2 = "</s>"

        prompt = conv.get_prompt()

        if args.model_type == "llama-2-chat":
            prompt += " "
    else:
        # conv.append_message(conv.roles[0], input_text)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt() + " "
        prompt = input_text + " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    cu_len = input_len
    totaltime = 0
    start_time = time.time()
    total_ids = 0
    prefill_time = 0.0
    new_tokens = 0
    step = 0


    if use_eaInfer:
        for output_ids in model.ea_generate(input_ids, temperature=temperature, top_p=top_p,
                                            max_steps=args.max_new_token):
            cost_time = (time.time() - start_time)
            if total_ids == 0:
                prefill_time = cost_time
            # totaltime += cost_time
            total_ids += 1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True,
                                          spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True,)
            cu_len = output_ids.shape[1]
            step += 1

            new_tokens = cu_len - input_len


        totaltime += (time.time() - start_time)

    else:
        for output_ids in model.naive_generate(input_ids, temperature=temperature,
                                               top_p=top_p,
                                               max_steps=args.max_new_token):
            cost_time = time.time() - start_time
            if total_ids == 0:
                prefill_time = cost_time
            total_ids += 1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            cu_len = output_ids.shape[1]
            step += 1
            new_tokens = cu_len - input_len
        totaltime += (time.time() - start_time)
    # print("case key: ", case_key)
    # print("step= ", step)
    # print("prefill_time= ", prefill_time, input_len / prefill_time)
    # print("decode_time ", totaltime - prefill_time, new_tokens / (totaltime - prefill_time))
    # print("new_tokens: ", new_tokens)
    # print("============================")
    return "{},{},{},{},{},{},{}\n".format(case_key, step, new_tokens,
                                             prefill_time, input_len / prefill_time,
                                             totaltime - prefill_time,
                                             new_tokens / (totaltime - prefill_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle/llama2chat/7B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/7B/",
                        help="path of basemodel, huggingface project or local path")
    parser.add_argument("--model-type", type=str, default="llama-2-chat",choices=["llama-2-chat","vicuna","mixtral"], help="llama-2-chat or vicuna, for chat template")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument('--out_res', type=str, default="")
    parser.add_argument("--useEaInfer", action="store_true")
    parser.add_argument('--max_iter', type=int, default=3000)
    args = parser.parse_args()

    dataset = load_dataset("ccdv/cnn_dailymail",
                           "3.0.0",
                           cache_dir="",
                           split="test")
    dataset_input_key = 'article'
    # dataset_output_key = 'highlights'
    # print(len(dataset))
    print(dataset[0])


    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    warmup(args, model)

    cnt = 0
    print("useEaInfer: ", args.useEaInfer)
    with open(args.out_res, 'w', encoding='utf-8') as fout:
        for d in tqdm(dataset):
            res = test(args, model, d[dataset_input_key], temperature=0.0, top_p=0.9,
                use_eaInfer=args.useEaInfer, use_sysm_prompt=False, case_key=d['id'])
            fout.write(res)
