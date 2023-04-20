---
layout: default
title: GPTQ A to Z 1
parent: Paper Review
nav_order: 1
---
Transformer 모델과 BERT와 GPT가 등장하면서, 언어모델의 발전은 무시무시할 정도의 속도로 현재까지 이어지고 있다. 급기야 더 이상 개인의 컴퓨팅 파워로는 감당하지 못할 수준의 65B, 175B 모델까지 등장하는 등, 발전의 방향은 획기적인 metric의 발견이 아니라 더 많은 데이터와 더 큰 사이즈로 학습시키는 방향으로 나아가기 시작했다.  

이에 대한 반대급부로 거대한 모델을 경량화하기 위해 Distillation, Pruning 등 다양한 기법이 등장하기 시작했다. 이번 논문인 GPTQ 또한 float16으로 저장되는 가중치를 4byte 이하의 더 작은 용량으로 *quantize* 하여 모델 경량화를 시도한 논문이다.  

개인적으로도 이러한 방향성에 대해서 긍정적인 이유가 있는데, 기존의 여러 컴퓨터 연산에 비해 DNN(Deep Neural Network)에서는 각 가중치에 대한 연산의 정확도가 비교적 덜 중요한 요소이기 때문이다. 컴퓨터의 실수 연산에 대한 오차로 인해 미사일 오폭 같은 치명적인 결과가 도래하는 것과 달리 여기선 오히려 정확도는 떨어지더라도 연산을 병렬적으로, 빠르게 진행할 수 있는 방법이 필요한 것이다.  

이 논문의 시작은 *Optimal Brain Surgeon*이라고, layer에서 중요성이 낮은 weight를 선정해 0으로 *pruning*하는 방법론이다. 그러므로 이 논문을 처음부터 끝까지 이해하기 위해서 이 방법론을 설명하고 있는 1993년의 논문부터 시작하고자 한다.  

<br>
<br>    

### Optimal Brain Sergeon and Greneral Network Pruning  

<br>  

전술했듯이, 이 논문은 layer에서 중요성이 낮은 weight를 선정해 이를 0으로 만든다. 0으로 만드는 것은 inference나 learning 시 0의 weight에서 연산을 skip하는 방법으로 연산량을 줄일 수 있다는 이점이 있다.  

이때 어떤 '중요성이 낮다'는 것을 판단할 수 있는 기준으로, 논문에서는 weight의 변화량($$\delta w$$)에 따른 Error의 변화량($$\delta E$$)을 사용한다.

$$
\delta E = E(w_0)-E(w_0+\delta w)
$$

이때 해석이 어려운 $$E$$ 대신 다음과 같이 **테일러 급수**를 이용해 근사하여 표현할 수 있다.

$$
\delta E = {\frac {\partial E(w_0)}{\partial w}}^T(w-w_0) + {\frac {1}{2}} (w-w_0)^T {\frac {\partial ^2 E(w_0)}{\partial w^2}} (w-w_0) + \cdots
$$

보통 2계 도함수 이후로는 오차항으로 두고 생각하지 않으므로, 0이라고 생각하자. 또한 현재 Network가 local minimum에 도달하였다는 가정 하에, $${\frac {\partial E(w_0)}{\partial w}}$$는 0이다. 그러므로 위의 테일러 급수는 다음과 같다.  

$$
{\begin{aligned}&\delta E = {\frac {1}{2}} \delta w^T · \mathbf{H} · \delta w\\&\mathbf{H}={\frac {\partial ^2 E(w_0)}{\partial w^2}}\end{aligned}}
$$

$$\delta w$$는 $$w$$에서 어떤 가중치를 pruning할 것인지를 결정하는 값이다. 예를 들어 pruning하고자 하는 q번째 weight를 $$w_q$$ 라고 정의하면, $$\delta w_q = -w_q$$이다. 좀 더 일반화하자면, $$q$$번째의 값이 1인 단위행렬 $$e_q$$를 정의하여 $$e_q^T·\delta w+w_q=0$$ 로 표현할 수 있다. 이것이 $$\delta E$$에서의 제약조건이 된다.  
**라그랑주 승수법**을 사용하면 이러한 제약조건 하에서의 극점 및 극값을 구할 수 있다.  

$$
{\begin{aligned}&{L = {\frac {1}{2}} \delta w^T · \mathbf{H} · \delta w + \lambda(e_q^T · \delta w+w_q)}\\&{\delta w=-{\frac {w_q}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}^{-1}·e_q}}\\&{L_q={\frac {1}{2}}{\frac {w_q^2}{[\mathbf{H}^{-1}]_{qq}}}}\end{aligned}}
$$  
  
$$[\mathbf{H}^{-1}]_{qq}$$는 $$\mathbf{H}^{-1}$$에서 $$q$$번째 행과 열의 값을 의미한다($$\mathbf{H}$$은 헤세 행렬이라 말하며, 대칭행렬이다.). 결국 **라그랑주 승수법**을 이용해 풀어낸 결과를 풀어서 설명하면 다음과 같이 표현 가능하다.
> $$w_q$$를 0으로 pruning할때, $$\delta w=-{\frac {w_q}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}^{-1}·e_q}$$ 에서 $$\delta E$$는 최솟값 $${\frac {1}{2}}{\frac {w_q^2}{[\mathbf{H}^{-1}]_{qq}}}$$을 가진다.  

남은 일은 모든 $$w$$ 내 가장 작은 $$\delta E$$ 를 가지는 $$q$$를 찾은 후, $$\delta w$$만큼 weight를 변경하는 것이다. 이 논문에서는 $$\delta E$$의 상한을 정해두고, 모든 $$q$$에 대해 상한 이하의 $$\delta E$$가 없을 때까지 이 과정을 반복한다.

<br>
<br>    

### Optimal Brain Compression: A Framework for Accurate post-Training Quantization and Pruning

<br>  

2022년에 나온 이 논문은 위의 프로세스를 어떻게 효율적으로 수행하고, 이를 *quantization*에 어떻게 활용할 수 있을지에 대해 논하고 있다.  
전체적으로는 어떤 가중치를 prining 할 것인지가 아니라 어떤 가중치를 '먼저' *quantization*할 것인지에 대한 기준을 OBS를 변형시킨 다음의 식으로 판별하고, 가중치를 변경하는 것이다.  

$$ w_p = argmin_{w_p} {\frac{(quant(w_p)-w_p)^2}{[\mathbf{H}^{-1}]_{pp}}}, \quad \delta_p=-{\frac{w_p-quant(w_p)}{[\mathbf{H}^{-1}]_{pp}}}·\mathbf{H}^{-1}_{:,p} $$  

이때 저자는 두 가지 아이디어를 적용해서 연산량을 줄이고 소요 시간을 단축하였다.  

 첫 번째는 매 pruning step마다 연산해야 하는 $$\mathbf{H}^{-1}=(2\mathbf{X}\mathbf{X}^T)^{-1}$$를 구하는 방법이다.  
 pruning된 가중치 값을 masking하는 pruning mask $$M$$에 대해 $$\mathbf{H}_M=2\mathbf{X}_M\mathbf{X}_M^T$$ 자체는 $$\mathbf{H}$$에서 $$M$$에 해당되는 열과 행을 0으로 만든 값이므로 구하기 어렵지 않지만, 문제는 $$\mathbf{H}_M^{-1}$$을 구하는 것이다. 가우스 소거법을 통해 역행렬을 구하는 연산은 $$O(n^3)$$의 시간복잡도가 소요되기에 매 step마다 이 연산을 수행하는 것은 상당한 Resource가 소모되는 일이다. 그래서 저자는 이미 연산된 $$\mathbf{H}^{-1}$$로부터 $$\mathbf{H}_M^{-1}$$를 구할 수 있는 방법에 대해 고안하였다.  

 아쉽게도 $$(\mathbf{H}_M)^{-1}≠(\mathbf{H}^{-1})_M$$ 이기에 해당 식을 만족시키기 위해선 우항에서 추가 연산이 필요하다. $$\mathbf{H}^{-1}\mathbf{H}=\mathbf{I}$$ 에서 $$\mathbf{H}^{-1}$$을 mask에 해당되는 $$p$$번째 행과 열을 모두 0으로 만들 수 있는 행렬변환 $$(\mathbf{I}-\mathbf{K})$$를 생각해보자. 조금만 생각해보면 행렬 $$\mathbf{A}$$는 다음과 같이 $$i$$열에서만 0이 아닌 값을 가짐을 짐작할 수 있다.  

 $$\mathbf{K}_{i,j}={\begin{cases}H^{-1}_{p,i}/{H^{-1}_{p,p}}&i \in d_{col},\\0&others\end{cases}}$$  

행렬 $$\mathbf{A}$$를 $$\mathbf{H}^{-1}\mathbf{H}=\mathbf{I}$$의 양 변에 곱하면 다음과 같은 꼴이 된다.  

$$[(\mathbf{I}-\mathbf{K})\mathbf{H}^{-1}][\mathbf{H}]=[\mathbf{I}-\mathbf{K}]$$  

$$\left[{\begin{matrix}\mathbf{A}_1 & \mathbf{0} & \mathbf{A_2} \\ \mathbf{0}^T & 0 & \mathbf{0}^T \\ \mathbf{A_3} & \mathbf{0} & \mathbf{A_4}\end{matrix}}\right]\left[{\begin{matrix}\mathbf{B_1} & \mathbf{b_1} & \mathbf{B_2} \\ \mathbf{b_2}^T & b_3 & \mathbf{b_4}^T \\ \mathbf{B_3} & \mathbf{b_5} & \mathbf{B_4}\end{matrix}}\right] = \left[{\begin{matrix}\mathbf{I} & \mathbf{K}_{:p,p} & \mathbf{0} \\ \mathbf{0}^T & 0 & \mathbf{0}^T \\ \mathbf{0} & \mathbf{K}_{p+1:,p} & \mathbf{I}\end{matrix}}\right]$$  

여기서 잘 생각해보면 $$H$$의 $$i$$번째 열과 행에 해당하는 $$\mathbf{b_n}$$들과 $$b_3$$은 $${(I-K)H^{-1}}$$의 영벡터 때문에 우항의 $$i$$열을 제외한 부분에 영향을 미치지 못함을 알 수 있다. 그러므로 $$\mathbf{b_n}$$들과 $$b_3$$을 모두 영벡터와 0으로 만들면 다음과 같다.  

$$\left[{\begin{matrix}\mathbf{A}_1 & \mathbf{0} & \mathbf{A_2} \\ \mathbf{0}^T & 0 & \mathbf{0}^T \\ \mathbf{A_3} & \mathbf{0} & \mathbf{A_4}\end{matrix}}\right]\left[{\begin{matrix}\mathbf{B_1} & \mathbf{0} & \mathbf{B_2} \\ \mathbf{0}^T & 0 & \mathbf{0}^T \\ \mathbf{B_3} & \mathbf{0} & \mathbf{B_4}\end{matrix}}\right] = \left[{\begin{matrix}\mathbf{I} & \mathbf{0} & \mathbf{0} \\ \mathbf{0}^T & 0 & \mathbf{0}^T \\ \mathbf{0} & \mathbf{0} & \mathbf{I}\end{matrix}}\right]$$  

$$[(\mathbf{I}-\mathbf{K})\mathbf{H}^{-1}]_{-p}[\mathbf{H}]_{-p}=\mathbf{I}_{-p}$$  

즉, $$(\mathbf{H}_{M})^{-1}=((\mathbf{I}-\mathbf{K})\mathbf{H}^{-1})_M$$이며, 이걸 논문에서는 아래와 같이 표현하고 있다.
$$\mathbf{H}^{-1}_{-p}=\left(\mathbf{H^{-1}}-{\frac{1}{[H^{-1}]_{pp}}\mathbf{H}^{-1}_{:,p}\mathbf{H}^{-1}_{p,:}}\right)_{-p}$$
이 결과를 이용하면 맨 처음 $$\mathbf{H}^{-1}$$를 구한 이후 pruning step마다 위의 식에 따라 $$\mathbf{H}^{-1}$$를 변경하면 $$O(n^2)$$의 시간복잡도로 OBS를 수행할 수 있다.  

저자는 각 행별로 pruning을 진행할 때 각 행에서 어떤 weight를 pruning할지 결정하는 것은 다른 행의 영향을 받지 않는다는 점에서 착안해 각 행의 연산을 병렬적으로 수행하였으며, weight 조정을 전체 행 단위로 한꺼번에 진행하여 속도를 늘릴 수 있었다고 두번째 아이디어를 설명하고 있지만, 요건 그렇게까지 와닫는 개선점은 아니다. 애초에 batch 단위로 모델을 학습시키고 inference하는 것은 기존에도 많이 쓰였던 방법이기도 하거니와, 어자피 행 단위로 헤세 행렬의 업데이트를 해야 하는 시점에서 전체 행렬의 weight update가 가능하다고 해서 유의미한 시간 단축이 가능할까 싶긴 하다. 내가 잘 이해하지 못하는 부분일수도 있겠다.

또한 pruning과 달리 *quantization*의 경우 모든 weight에 적용하는 거여서, weight를 업데이트하는 데 몇 가지 전략을 수정해야만 했다. 저자는 오히려 나머지 weight 조정이 가능한 초기 단계에 높은 loss를 가지는 weight를 먼저 *quantization*하는 것이 오히려 전체 성능을 더 올릴 수 있었다고 한다. 그렇게 생각하면 OBS에서 제시하는 Greedy한 알고리즘이 *quantization*에서는 그다지 큰 의미를 가지지 않는 게 아닐까 싶기도.


<br>
<br>    

### GPTQ: Accurate Post-Training Quantization For Generative Pre-Trained Transformers
<br> 

이제야 원 논문에 대해 소개할 수 있게 되었다. 이 논문은 이전 논문들에서 연구한 결과를 바탕으로 GPT 등의 대규모 LM의 *quantization*을 진행하였다. 100B이 넘어가는 대규모의 transformer model에서 *quantization*을 수행 가능하려면 더 많은 시간 단축이 필요한데, 본 논문은 헤세 행렬의 업데이트를 과감하게 생략하는 방법을 고안하였다.  

이 아이디어의 시작은 이전 논문에서부터 이어져오는 의문점인데, 전체 weight를 업데이트 하는 *quantization*의 특성상 낮은 loss를 가지는 weight를 먼저 수정하는 Greedy 알고리즘은 결국 높은 loss를 가지는 weight를 가중치 조정으로 어느 정도 완충이 가능한 weight가 거의 없는 후반부에 수정하게 됨으로써 오히려 최적화에서 멀어지는 결과를 낳는다는 것이다.  
본 논문에서는 그럴 거면 차라리 각 row에서 처음부터 weight를 순차적으로 *quantization*하는 게 낫겠다고 제안한다. 그렇게 하더라도 이전의 방법과 비교해서 큰 loss가 발생하는 건 아니었다고 한다.  

순차적으로 *quantization*하는 방법을 선택하게 되면 각 step별로 $$\mathbf{H}^{-1}$$을 업데이트하는 과정을 생략할 수 있는 기발한 아이디어가 있는데, 바로 $$\mathbf{H}^{-1}$$의 **숄레스키 분해** 결과를 활용하는 것이다.  
어떤 수학적 공통점이 있는지는 잘 모르겠지만, $$\mathbf{H}^{-1}$$를 첫 index부터 차례대로 pruning하면서 업데이트하면서 나타나는 값은 $$\mathbf{H}^{-1}$$를 숄레스키 분해한 결과(상삼각행렬)의 각 행의 값에서$$([\mathbf{H}^{-1}_{F_q}]_{qq})^{1/2}$$로 나눈 값과 동일하게 나타난다. 그럼 각 step마다 $$\mathbf{H}^{-1}$$를 업데이트 하는 데 드는 $$O(n^2)$$의 시간복잡도를 없앨 수 있으므로, 어느 정도 loss를 감수하는 대신(사실 Greedy 알고리즘이 최선이 아님을 아는 시점에서 정말 loss가 발생했는지도 확실하지 않지만) 매우 빠른 연산 속도로 LLM의 *quantization*이 가능하다.  

또한 연산량에 비해 수많은 weight를 update하는 I/O 과정에서 병목이 발생하는 점을 해결하기 위해 'Lazy Bach-Updates', 128개의 batch에 대해 batch 안의 weight는 각 step마다, 그리고 batch 이후의 weight는 batch가 종료된 후 한꺼번에 업데이트하는 식으로 연산량과 I/O의 균형을 맞추고자 했다. 실제로 30B의 Llama 모델을 *quantization*하는 데 하나의 3090으로 3시간 가량이 걸림을 확인할 수 있었다.  

연산량을 줄여 개인 단위에서도 *quantization*을 수행할 수 있게 된 것은 확실한 진전이나, 결국 이런 순차적인 논문의 흐름은 OBS의 pruning weight를 선택하는 Greedy 알고리즘을 적용하는 것이 모든 weight를 *quantize*하는 경우 좋은 생각이 아님을 보여주는 것 같다. 어떤 weight를 순차적으로 변경시키는 것 말고 다른 방향에서 *quantizaiton*을 수행할 수 있는 방법이 무엇일까에 대한 생각을 해보게 된다.

