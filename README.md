# chatKoAlpaca
KoAlapaca + Chat

훌륭한 LLM KoAlpaca에 chat 튜닝을 할 수 있습니다.

제가 작업한 환경은 A100 80G 이며 대부분의 LLM에서 CUDA OOM이 발생합니다.

따라서 BASE LLM은 EleutherAI/polyglot-ko-1.3b 이 모델 입니다.

베이스 모델을 Alpaca 데이터셋으로 Finetune을 진행하고,

Alpaca 모델을 ColossalAI를 활용하여 chat Finetune하는 과정을 거칩니다.

Alpaca base는 [KoAlpaca](https://github.com/Beomi/KoAlpaca) 이며,

chat Finetune 과정은 [KoChatGPT](https://github.com/airobotlab/KoChatGPT) 을 base로 하였으며, 

ColossalAI는 현재 사용한 gpt-neox 기반의 모델을 지원하지 않아 [KoChatGPT-수정](https://github.com/HaloKim/KoChatGPT.git) 을 수정하여 적용하였습니다.

# Pipeline

![image](https://github.com/HaloKim/chatKoAlpaca/assets/44603549/7affcdb9-b566-42fc-838f-f6012edeae7e)

# 학습

```
Episode [1/1]:  67%|██████▋   | 2/3 [00:24<00:12, 12.21s/it]
Train epoch [1/1]:   0%|          | 0/3 [00:00<?, ?it/s]
Train epoch [1/1]:   0%|          | 0/3 [00:00<?, ?it/s, actor_loss=0, critic_loss=0]
Train epoch [1/1]:  33%|███▎      | 1/3 [00:00<00:01,  1.78it/s, actor_loss=0, critic_loss=0]
Train epoch [1/1]:  33%|███▎      | 1/3 [00:01<00:01,  1.78it/s, actor_loss=0, critic_loss=0]
Train epoch [1/1]:  67%|██████▋   | 2/3 [00:01<00:00,  1.88it/s, actor_loss=0, critic_loss=0]
Train epoch [1/1]:  67%|██████▋   | 2/3 [00:01<00:00,  1.88it/s, actor_loss=0, critic_loss=0]
Train epoch [1/1]: 100%|██████████| 3/3 [00:01<00:00,  1.91it/s, actor_loss=0, critic_loss=0]
Episode [1/1]: 100%|██████████| 3/3 [00:37<00:00, 12.64s/it]
```

# 결과

```
######################################################################
Below is an instruction that describes a task.
아래는 작업을 설명하는 명령어입니다.

Write a response that appropriately completes the request.
명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction(명령어):
불고기용 고기 한우에요?

### Response(응답):'저는 인공지능 언어 모델로, 물리적 개념인 소고기와 돼지고기는 다른 종류이며, 불고그용 고기 역시 일반적으로 사용되는 종류가 아닙니다. 따라서, 질문에 대한 정확한 답변은 제공할 수 없습니다. 😊 😊 😊 😊죄송합니다. 😊하지만, 한우와 돼지고기를 사용하여 불고기를 만들 수 있는 방법이 있습니다. 일반적으로 불고기 요리를 할 때는 소고기를 사용하지만, 한우와 돼지고기도 불고기 요리에 사용될 수 있습니다. 이들 고기를 이용해 불고기와 된장
######################################################################
Below is an instruction that describes a task.
아래는 작업을 설명하는 명령어입니다.

Write a response that appropriately completes the request.
명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction(명령어):
리처드 닉슨이 43대 부통령직을 수행한 년도는?

### Response(응답):'리처드 닉슨이 43대 부통령직을 수행한 년도는 1978년입니다.인 1978년 11월에 닉슨은 43대 부통령으로 지명되었습니다.당한 년도는 1978년입니다.은 닉슨의 업적을 인정하지 않는 일부 언론에 대한 탄압과 닉슨의 사생활과 관련된 문제를 제기하면서 부통령직에서 물러났습니다.은 닉슨의 재임기간 중 그의 업적을 인정하는 일부 언론에 대한 탄압과 사생활과 관련된 문제를 제기하면서 부통령직에서 물러났습니다.은 닉슨의 업적을
######################################################################
Below is an instruction that describes a task.
아래는 작업을 설명하는 명령어입니다.

Write a response that appropriately completes the request.
명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction(명령어):
시카고 오헤어 국제공항은 어디에 있어

### Response(응답):'시카고 오헤어 국제공항은 미국 일리노이주 시카고 오헤어 국제공항에 위치해 있습니다.. 국제공항에서도 항공편이 있습니다.시카고 오헤어 국제공항은 미국 시카고에 위치한 공항으로, 시카고에서 약 3시간 정도 소요됩니다.시카고 오헤어 국제공항은 미국의 시카고에 위치한 국제공항으로, 시카고에서 약 1시간 정도 소요됩니다.시카고 오헤어 국제공항은 미국 시카고에서 3시간 정도 소요됩니다.시카고 오헤어 국제공항은 미국 시카고 오헤어 국제공항에서 약 30분 정도 소요됩니다.시카고 오헤어 국제공항은 미국 일리노이주 시카고에 위치해 있
######################################################################
Below is an instruction that describes a task.
아래는 작업을 설명하는 명령어입니다.

Write a response that appropriately completes the request.
명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction(명령어):
오늘 미세먼지 어때?

### Response(응답):'저는 인공지능 챗봇이므로 미세먼지 농도가 어떻게 발생하는 지에 대한 정보를 알지 못합니다. 하지만 일반적으로 미세먼지 농도는 지역에 따라 다르며, 특히 황사나 미세먼지 등의 오염 물질은 중국발 미세먼지 등 다양한 요인에 의해 발생할 수 있습니다. 따라서 해당 지역의 기상 예보나 대기 상태를 참고하시는 것이 좋습니다.... 예보, 기상특보 등을 확인하여 대응하시기 바랍니다.. 👌.. 예보, 기상특보 등은 인터넷이나 해당 지역의 기상청 등에서 확인하실 수 있습니다.. 👌
```

# Reference

[KoAlpaca](https://github.com/Beomi/KoAlpaca)

[EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)

[ColossalAI](https://github.com/hpcaitech/ColossalAI)

[KoChatGPT](https://github.com/airobotlab/KoChatGPT)

[KoChatGPT-수정](https://github.com/HaloKim/KoChatGPT.git)
