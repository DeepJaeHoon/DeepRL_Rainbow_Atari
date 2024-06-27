과제 내용 : gymnasium의 Atari breakout을 강화학습시키기

사용한 알고리즘 - Rainbow DQN

최고점수 - 423점

학습환경 - RTX 3060, 학습하는데 3~4일정도 걸린거로 기억해요

1. 가중치에서 숫자는 Save 단위입니다. 
 100이라면 score가 100 ~ 200 사이일 때 저장되는 값입니다.

2. max는 최댓값 달성하면 저장되는 가중치이며 period는 5 에피소드마다 저장되는 가중치입니다.

3. 학습용 코드는 RAINBOW_DQN, 영상 녹화용은 RAINBOW_DQN_VIDEO_RECODER입니다.
   가중치는 period 가중치를 사용했습니다.

4. V5 환경에서 학습했습니다.


https://github.com/DeepJaeHoon/DeepRL_Rainbow_Atari/assets/174041317/879bed1a-30c3-48d4-8a5c-78e4464c0239

