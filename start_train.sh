autotrain llm --train --project_name my-llm --model mistralai/Mistral-7B-v0.1 --data_path . --use_peft --use_int4 --target-modules q_proj,v_proj --learning_rate 2e-4 --train_batch_size 12 --num_train_epochs 3 --trainer sft