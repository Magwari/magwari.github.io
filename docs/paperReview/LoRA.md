---
layout: default
title: LoRA
parent: Paper Review
nav_order: 2
---

이번에는 LoRA 논문을 리뷰해보자 한다. 논문이 나온지는 꽤 되었고, 실제로 이 방법을 이용해 학습한 모델도 많이 나왔다. 아마 rtx 3090만으로 모델 개발을 해야 하는 비루한 나의 컴퓨터로는 30, 65b의 모델을 돌리거나 학습하기 위한 방법 중 하나로 쓸 수 있지 않을까 싶다.
아이디어 자체는 그리 새로울 것은 없다. 모델을 구성하는 weight matrix를 fix하고, 별도의 matrix를 initialize하여 새롭게 학습하는 구조인데, 이런 방법은 몇년 전 bert가 유행했을 때에도 한두번씩 보였다. 나 또한 졸업논문으로 topic modeling의 embedding vector를 이용하여 bert 가중치를 update하는 비슷한 방법론을 사용했던 적이 있고.  
그래서 이 논문에서는 모델 구조보다는 실제 모델을 어떻게 학습시켰는지에 대한 실증적인 측면을 더 살펴볼 예정이다.

<br>  
<br>

### LoRA: Low-Rank Adaption of Large Language Model

<br>

모델을 구성하는 Pretrained Weight마다 별도의 
$$\mathbf{A} \in \mathbb R^{(d \times r)}$$, $$\mathbf{B} \in \mathbb R^{(r \times d)}$$
를 initialize하여 
$$\mathbf{A}$$, $$\mathbf{B}$$
만 학습하는 것이 기본 골조다. 이때 $$r$$를 rank라고 하는데, 이 $$r$$값을 1, 2, 4, 8과 같이 $$d$$에 비해 매우 작게 만들어 AutoEncoder와 비슷한 형태의 Layer로 만든다.

논문에서는 다양한 모델에 대해 LoRA를 적용했다. 기존의 많은 prior works와 비교했을 때 LoRA는 상당한 성능 개선을 보였으며, 일부 데이터셋에 대해서는 모델 자체를 Finetuning한 것보다 좋은 성능을 냈다고 한다.  

논문은 모델 적용을 위한 hyper parameter를 어떻게 설정하는지에 대해 실험 내용을 포함하고 있는데, "어떤 Weight에 LoRA를 적용할지"와 "Optimal $$r$$이 얼마인가"에 대한 내용이다. 결론적으로는 Transformer에 존재하는 모든 Weight (
$$W_q, W_k, W_v, W_o$$ 
)에 대해 $$r=2$$로 적용하는 것이 성능이 가장 좋았다.  
그러니까 rank는 매우 낮아도 관계가 없으니, 모든 Matrix에 대해 LoRA를 적용하는 것이 중요하다는 것으로 해석된다.

<br>  

### ※ 낮은 $$r$$로도 문제가 없나?  

<br>

논문에서는 $$r$$를 낮게 잡아도 괜찮은 이유를 찾기 위해 학습된 $$\mathbf{A}$$로 만들어지는 Vector Space의 Subspace간 유사도를 $$r$$의 변화에 따라 구한 결과를 비교하였다. 이때 
**그라스만 거리(Grassman distance)** 결과를 활용했는데, 구하는 공식은 아래와 같다.  

$$
\phi(A_{r},A_{r'},i,j) = \frac{||U^{i \mathsf{T}}_{r}U^{j \mathsf{T}}_{r'}||^2_F}{min(i,j)}
$$  

(더 자세한 식은 논문의 Appendix G. 를 참고)

이 값은 0에서 1 사이의 값을 가지는데, 클수록 두 벡터의 거리가 가깝다는 것을 의미한다. 논문에서는 $$r=8, r'=64$$ 두 경우를 비교했는데, 첫 Subspace($$i,j=1$$)의 유사도가 0.5 이상임을 보여주었고, 이를 $$r$$이 극도로 낮음에도 불구하고 좋은 성능을 보여준 이유라고 하였다.  

또 다른 방법으로 고정된 가중치 $$W$$를 $$\Delta W$$의 $$r$$차원 Subspace로 Project하는 방법을 사용했는데, 이때 $$W$$, $$\Delta W$$의 특이값 분해를 통해 얻은 $$\mathbf{U}, \mathbf{V}$$를 활용하였다고 한다. 비교분석을 위해 random matrix를 $$\mathbf{U}, \mathbf{V}$$로 한 결과 또한 계산하였다.  

만약 $$W$$와 $$\Delta W$$의 "강조하고자 하는 direction"이 같다면 project된 vector $$U^\mathsf{T} W V^\mathsf{T}$$의 Frobenius norm의 값이 커질 것이고, 아니면 작아질 것이다. 논문에서는 이 값으로 $$\Delta W$$ 자체의 Frobenius norm을 나눠서 Amplification Factor를 구했고, "task-specific direction으로 얼마나 많이 강조되었는지"를 측정하는 지표로 사용하였다. 계산 결과 r이 크면 클수록 Amplification Factor는 감소함을 확인할 수 있었다.

논문은 이를
1. $$\Delta W$$는 $$W$$와 비교적 강한 연관성을 가지며, $$W$$의 특정 feature를 증폭하는 효과를 가진다.
2. $$\Delta W$$ 는 $$W$$가 "강조하지 않은 direction을 강조"한다.  

라며, $$r$$이 낮을수록 이러한 "강조"효과가 더 크다고 설명하였다. 그래서 낮은 $$r$$에 대해서도 성능이 잘 나올 수 있었다고.  

LoRA의 아이디어 자체는 새로운 breakthrough를 마련했다 정도로 평가하지는 않지만, 개인적으로 Subspace를 이용한 검증 metric은 눈여겨볼만한 점이 많은 것 같다. 특이값 분해에 대해 다시 한번 더 복습해보면서 singular vector에 대해 직관적으로 이해할 수 있다면, 모델 학습에 대한 평가 지표로 활용할 수 있는 방안을 찾을 수 있을지도 모르겠다.