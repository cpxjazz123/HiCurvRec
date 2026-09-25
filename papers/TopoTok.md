## Topology-Aware Tokenization for Generative Recommendation

Yaokun Liu University of Illinois Urbana-Champaign Champaign, United States yaokunl2@illinois.edu Yifan Liu University of Illinois Urbana-Champaign Champaign, United States yifan40@illinois.edu Zhenrui Yue University of Illinois Urbana-Champaign Champaign, United States zhenrui3@illinois.edu Gyuseok Lee University of Illinois Urbana-Champaign Champaign, United States gyuseok2@illinois.edu

Zelin Li University of Illinois Urbana-Champaign Champaign, United States zelin3@illinois.edu Ruichen Yao University of Illinois Urbana-Champaign Champaign, United States ryao8@illinois.edu Dong Wang University of Illinois Urbana-Champaign Champaign, United States dwang24@illinois.edu

Figure 1: Illustration of topology distortion in item tokenization. Left: Item adjacency relationships in the continuous embedding space are compromised after RQ-VAE quantization. Right: Top-20 neighbor overlap between continuous and discrete spaces decreases with quantization depth.

<!-- image -->

## 1 INTRODUCTION

The advancement of Large Language Models (LLMs) has catalyzed a paradigm shift in sequential recommendation, moving from traditional embedding-based retrieval [4, 17, 29, 41] to generative recommendation [23, 28, 40, 43]. By reframing sequential recommendation as an autoregressive generation task, LLMs can directly generate target item identifiers, effectively bypassing the scalability bottlenecks and retrieval overhead of traditional Approximate Nearest Neighbor (ANN) search in high-dimensional spaces [28]. Within the generative recommendation framework, item tokenization plays a critical role by transforming continuous item semantics into discrete identifiers that serve as generation targets, and its quality directly governs the model's ability to perceive item relationships and the ultimate accuracy of generative predictions.

Early item tokenization methods fall into two categories: (1) Pseudo ID-based methods , which assign each item a unique identifier via techniques such as random indexing [9, 13, 43], offering high efficiency but lacking semantic information [30]; and (2) text-based methods , which replace item IDs with textual content (e.g., titles or descriptions) and formulate interactions as natural language prompts [2, 11, 13], capturing rich semantics but incurring high computational cost and potentially producing invalid outputs [13].

To balance the semantic richness of text-based representations and the efficiency of numerical IDs, recent work has focused on semantic ID-based item tokenization [28], which maps continuous

## Abstract

Generative recommendation reformulates sequential recommendation as an autoregressive generation task, yet a critical issue in this paradigm remains overlooked: topology distortion in item tokenization. In particular, we observe that the intrinsic adjacency relationships of items in the pretrained semantic embedding space are significantly disrupted after quantization. This topology distortion misleads the model's perception of item similarity, ultimately bottlenecking the accuracy of generative recommendations. To address this issue, we propose Topo logy-Aware Tok enization ( TopoTok ), an item tokenization framework that preserves item relational structure throughout the quantization hierarchy. Different from the prior monolithic supervision in tokenization, TopoTok introduces a multi-level distillation scheme to progressively recover the topology from coarse to fine granularity: 1) Inter-Group Distillation to capture global cluster-wise relations; 2) Intra-Group Distillation to refine local structures within semantic clusters; and 3) Inter-Item Distillation to enforce fine-grained alignment at the individual item level. Extensive experiments on three benchmark datasets demonstrate that TopoTok effectively alleviates topology distortion and consistently outperforms state-of-the-art tokenizers, achieving significant performance gains of up to 9.42% in Recall@5.

## CCS Concepts

· Information systems → Recommender systems .

## Keywords

Generative Recommendation, Sequential Recommendation, Item Tokenization, Knowledge Distillation, Topology Preservation

## ACMReference Format:

Yaokun Liu, Yifan Liu, Zhenrui Yue, Gyuseok Lee, Zelin Li, Ruichen Yao, and Dong Wang. 2026. Topology-Aware Tokenization for Generative Recommendation. In 20th ACM Conference on Recommender Systems (RecSys '26), September 27-October 02, 2026, Minneapolis, MN, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3773078.3831780

<!-- image -->

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0)

RecSys '26, Minneapolis, MN, USA © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2284-4/2026/09 https://doi.org/10.1145/3773078.3831780

item embeddings into discrete token sequences, preserving semantic structure for autoregressive generation. Within this paradigm, the Residual Quantized Variational Autoencoder (RQ-VAE) [18] has become a dominant tokenizer due to its end-to-end learnability and ability to capture complex semantics. RQ-VAE employs a multi-layer codebook that hierarchically quantizes residual representations, encoding items from coarse to fine granularity. For example, in Figure 2, a digital piano is tokenized into &lt; 6 , 3 , 5 &gt; , corresponding to instruments, keyboards, and digital pianos.

However, we identify a critical topology distortion problem that emerges from semantic ID-based tokenization workflows, where we define topology as the relational structure among item representations induced by pairwise similarities (e.g., neighborhood ranking in the embedding space). While continuous item embeddings provide rich semantic information for recommendation by capturing the intrinsic relational structure among items, this essential knowledge is progressively disrupted during the residual quantization process. As illustrated in Figure 1, our empirical analysis reveals that for the top-20 neighbor relationships defined in the continuous embedding space, only 63% are preserved at layer 1, and this proportion further drops to 27% at layer 3 during RQ-VAE tokenization. Such distorted relational structures introduce "topological noise" into the item identifiers, where semantically non-neighboring items are mistakenly treated as similar, fundamentally compromising the LLM's perception of item relationships and degrading autoregressive prediction accuracy.

In this paper, we adopt topology distillation to transfer relational structure from the semantic embedding space to the item token space, directly addressing topology distortion by preserving item relationships. However, existing relational knowledge distillation methods are not directly applicable in this setting, as they often assume homogeneous continuous embedding spaces, whereas topology preservation under discrete quantization introduces fundamentally different constraints [16]. Specifically, effective distillation within the residual quantization hierarchy faces two key challenges: ( C1 ) Layer-wise supervision. The hierarchical architecture of RQVAE requires supervision to be applied at multiple layers; otherwise, a monolithic signal leads to supervision entanglement, where early layers are undertrained and deeper layers overfit to compensate for accumulated distortion. ( C2 ) Granularity alignment. Since RQ-VAE encodes semantics from coarse to fine, the distillation granularity must align with each layer's semantic role; enforcing fine-grained constraints uniformly can hinder early layers from capturing broader semantic structures.

Tothis end, we propose Topo logy-Aware Tok enization ( TopoTok ), a hierarchical tokenization framework with layer-aligned topology supervision that preserves topological fidelity throughout RQVAE-based item tokenization. Specifically, we propose a layer-wise supervision scheme that decomposes topology distillation into multi-level objectives that can be integrated into the RQ-VAE hierarchy ( C1 ) with aligned coarse-to-fine semantic granularity ( C2 ). Specifically, TopoTok utilizes three levels of topology distillation: (1) Inter-Group Distillation captures coarse-grained topology at early layers by aligning similarities between group-level representations; (2) Intra-Group Distillation refines local structure at intermediate layers by preserving relations within semantic groups; (3) Inter-Item Distillation enforces fine-grained alignment at the final layer via item-level similarity matching. This modular design allows TopoTok to be flexibly adapted to RQ-VAE architectures with an arbitrary number of quantization layers by mapping these distillation levels to the corresponding stages of semantic refinement. Collectively, the hierarchical supervision of TopoTok empowers each quantization layer to distill topological priors at a layer-specific granularity, effectively mitigating topology distortion across the codebook and ensuring that discrete token identifiers remain structurally consistent with the original semantic manifold. Extensive experiments on three benchmarks demonstrate that TopoTok improves topology preservation and outperforms existing item tokenization methods for generative recommendation.

## 2 RELATED WORK

## 2.1 Generative Recommendation

In recent years, generative recommendation has emerged as a paradigm that formulates sequential recommendation as an autoregressive generation task. In contrast to traditional embedding-based retrieval methods, which typically rely on a two-tower model to compute ranking scores followed by efficient MIPS or ANN search for top-k retrieval [8, 12, 15, 25], generative recommenders leverage the context understanding capability of LLMs to generate the identifier tokens of the next item directly, enabling more flexible modeling and better handling of challenges such as cold start [6]. Early effort P5 [9] fine-tunes a pretrained language model T5 [27] to handle multiple recommendation tasks in a single model. Another seminal work is TIGER [28], which proposes representing each item by a sequence of discrete semantic codes derived from item side information and then using a pretrained T5 to predict the next item's codes. Subsequent research has enhanced the generative recommendation by integrating more signals and improving training strategies. For example, EAGER [35] employs a two-stream generation framework to incorporate both user behavior history and item content semantics. ED 2 [36] leverages an end-to-end unified framework that integrates semantic and collaborative filtering indexes using a multi-grained token regulator and task-specific instruction tuning. Existing generative recommenders often overlook topological supervision at the item tokenization stage, resulting in item token representations with distorted item relations. Our work addresses this gap by integrating topological distillation supervision into the item tokenization process.

## 2.2 Item Tokenization

A key challenge in generative recommendation is designing item tokenizations that LLMs can both interpret and generate effectively. Existing approaches fall into three main categories: pseudo IDbased [3, 9, 13, 34], text-based [1, 5, 19-21, 37, 39], and semantic ID-based [28, 40]. Pseudo ID-based methods, such as P5 [9], assign unique tokens via techniques like random indexing, which are efficient but fail to capture intrinsic item-relatedness. Text-based methods [2] utilize item metadata to reformulate recommendations as instruction-following tasks. While expressive, text-based methods incur high computational costs and are prone to hallucination [13]. To bridge these gaps, semantic ID-based methods quantize item embeddings into discrete codes, preserving semantics in a structured form. For example, TIGER [28] pioneered the use of RQ-VAE [18]

as a backbone, leveraging its hierarchical codebook to encode items from coarse to fine granularity. Since its inception, RQ-VAE has emerged as the predominant framework for semantic ID tokenization. LETTER [33] integrates collaborative signals and diversity regularizations into tokenization. CoST [42] introduces contrastive loss to maintain neighborhood relationships but supervises only the final output, leading to entangled constraints across quantization layers. ETEGRec [22] exploits the end-to-end learnability of RQ-VAE to jointly optimize the tokenizer and the recommender. In industrial contexts, OneRec [7] alternatively uses RQ-KMeans to offer lightweight quantization. However, such non-parametric methods often struggle to capture the complex dependencies inherent in item semantics and lack the differentiability required for gradient-based joint optimization. In contrast, our work harnesses the differentiability of RQ-VAE to align topology distillation with the hierarchical semantic progression of residual quantization. This ensures that topological priors are systematically integrated into each stage of the codebook hierarchy in a coarse-to-fine manner.

## 3 METHODOLOGY

## 3.1 Preliminary

3.1.1 Problem Formulation. We formulate the generative recommendation task under the sequential recommendation scenario. Given the set of items I and a user interaction sequence S 𝑢 = [ 𝑖 1 , 𝑖 2 , . . . , 𝑖 𝑡 - 1 ] ∈ I , the task is to predict the next item 𝑖 𝑡 ∈ I . Generative recommendation addresses this task through two key steps: item tokenization and autoregressive generation. Item tokenization mapseachitem 𝑖 ∈ I into a token sequence c 𝑖 = [ 𝑐 𝑖, 1 , 𝑐 𝑖, 2 , . . . , 𝑐 𝑖,𝐿 ] ∈ C , where 𝐿 is the sequence length and C is predefined token set. The user interaction sequence S 𝑢 is thereby transformed into a tokenized sequence X 𝑢 = [ c 𝑖 1 , c 𝑖 2 , . . . , c 𝑖 𝑡 - 1 ] . Given X 𝑢 , the model autoregressively generates the token sequence c 𝑖 𝑡 of the next item 𝑖 𝑡 by factorizing the conditional probability as:

$$p ( c _ { i _ { t } } | X ^ { u } ) = \prod _ { l = 1 } ^ { L } p ( c _ { i _ { t } , l } | X ^ { u } , c _ { i _ { t } , 1 } , \dots , c _ { i _ { t } , l - 1 } ) . \quad ( 1 ) \\ \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

3.1.2 RQ-VAE for Semantic ID Tokenization. Semantic ID-based item tokenization maps continuous item embeddings into discrete numerical tokens while preserving their semantics. RQ-VAE is widely adopted as the backbone model for hierarchical semantic encoding through multi-layer residual quantization.

Given an item 𝑖 , we first obtain its embedding s 𝑖 ∈ R 𝑑 𝑠 with pre-trained encoders such as SASRec [17] or LLaMA [32], which captures collaborative or textual signals of items. The embedding s 𝑖 is then projected into a latent space by an MLP encoder:

$$z _ { i } = \text {Encoder} ( s _ { i } ) , \ \ z _ { i } \in \mathbb { R } ^ { d _ { c } } ,$$

The latent representation z 𝑖 is then quantized by a sequence of 𝐿 hierarchical codebooks {C 1 , . . . , C 𝐿 } , where each codebook C 𝑙 contains 𝑁 learnable code embeddings { e 𝑙,𝑗 ∈ R 𝑑 𝑐 } 𝑁 𝑗 = 1 . The quantization at layer 𝑙 is performed through recursive residual mapping:

$$c _ { l } = \arg \min _ { j } \| r _ { l - 1 } - e _ { l , j } \| ^ { 2 } , \quad r _ { l } = r _ { l - 1 } - e _ { l , c _ { l } } , \quad ( 3 )$$

with r 0 = z 𝑖 . Here, r 𝑙 - 1 denotes the residual from the ( 𝑙 - 1 ) -th layer, and 𝑐 𝑙 ∈ { 1 , . . . , 𝑁 } is the selected code index at layer 𝑙 . This recursive process yields a semantic ID c 𝑖 = [ 𝑐 𝑖, 1 , . . . , 𝑐 𝑖,𝐿 ] for item

𝑖 . The reconstructed latent representation ˆ z 𝑖 = 𝐿 𝑙 = 1 e 𝑙,𝑐 𝑙 is then decoded back to the embedding space:

$$\hat { s } _ { i } = D e c d o r ( \hat { z } _ { i } ) , \ \hat { z } _ { i } \in \mathbb { R } ^ { d _ { s } } .$$

The codebooks are optimized by minimizing the joint reconstruction and commitment loss:

$$\mathcal { L } _ { R Q \text {-VAR} } = \mathcal { L } _ { r e c o n } + \mathcal { L } _ { c o m m i t } ,$$

$$\mathcal { L } _ { r e c o n } = \| \hat { s } _ { i } - s _ { i } \| ^ { 2 } ,$$

$$\mathcal { L } _ { \text {commit} } = \sum _ { l = 1 } ^ { L } \| s g ( r _ { l - 1 } ) - e _ { l , c _ { l } } \| ^ { 2 } + \mu \| r _ { l - 1 } - s g ( e _ { l , c _ { l } } ) \| ^ { 2 } , \quad ( 7 ) \\$$

where sg (·) denotes the stop-gradient operation. The reconstruction loss L recon ensures the reconstructed embedding retains the original semantics, and the commitment loss L commit encourages proximity between the latent residuals and their assigned code embeddings.

## 3.2 Topology-Aware Tokenization (TopoTok)

As shown in Figure 2, we present TopoTok, which aligns topology supervision with the coarse-to-fine semantic progression of RQ-VAE by decomposing the distillation objective into three hierarchical levels: Inter-Group, Intra-Group, and Inter-Item.

3.2.1 Topology Distillation Formulation . In this section, we establish a mathematical framework for topology distillation. Formally, let { h t 𝑖 } 𝑀 𝑖 = 1 and { h s 𝑖 } 𝑀 𝑖 = 1 denote the representations of 𝑀 topological units in the teacher and student spaces, respectively. A topological unit defines the basic entity for relational alignment. To quantify the relational structure between units, we construct pairwise distance matrices D t , D s ∈ R 𝑀 × 𝑀 , where each entry 𝑑 𝑖 𝑗 measures the proximity between units 𝑖 and 𝑗 in the corresponding space. However, under quantization, information compression breaks the consistency of pairwise distances across spaces, making distance-level alignment ill-posed. Therefore, we formulate topology distillation as a similarity ranking alignment problem, which focuses on preserving relative neighborhood structure. Specifically, we transform distances into similarity distributions by applying a row-wise softmax to the negated distance matrices:

$$\det \quad P ^ { t } [ i , \colon ] = \text {softmax} ( - D ^ { t } [ i , \colon ] ) , \quad P ^ { s } [ i , \colon ] = \text {softmax} ( - D ^ { s } [ i , \colon ] ) . \ \ ( 8 )$$

Here, each row of P t and P s encodes the relative ranking of other units with respect to unit 𝑖 in the teacher and student spaces, respectively. By emphasizing neighbor ranking consistency rather than exact distance values, this formulation enables topology comparison across heterogeneous representation spaces.

Finally, topology distillation is achieved by minimizing the KL divergence between the teacher and student similarity distributions:

$$\mathcal { L } _ { \text {TD} } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } K L \left ( \mathbf P ^ { \text {t} } [ i , \colon ] \right \| \mathbf P ^ { s } [ i , \colon ] \right ) ,$$

which encourages the student representations to preserve the neighborhood ranking structure defined in the teacher space, thereby maintaining topological structure during the tokenization.

3.2.2 Multi-Granularity Topology Supervision . This section details how TopoTok conducts topology supervision across multiple semantic granularities to achieve coarse-to-fine topology distillation that aligns with the hierarchical structure of RQ-VAE.

Figure 2: Illustration of TopoTok. TopoTok preserves topological relationships in semantic ID-based tokenization by imposing hierarchical, coarse-to-fine topology distillation across residual quantization layers. A three-layer RQ-VAE is shown for illustration, while TopoTok generalizes to arbitrary depths.

<!-- image -->

Inter-Group (IG) . Inter-group level focuses on preserving coarsegrained topology at the group level, ensuring that high-level relationships among item groups in the continuous embedding space are distilled in the early-stage tokens learned by RQ-VAE.

Given a residual layer 𝑙 , each item 𝑖 is assigned to a semantic group based on its code index 𝑐 𝑖,𝑙 (Eq. 3). For each code 𝑗 ∈ { 1 , . . . , 𝑁 } at layer 𝑙 , we define the corresponding item group as:

$$\mathcal { G } _ { l , j } = \{ i \ | \ c _ { i , l } = j \} , \quad & ( 1 0 ) \\$$

which encapsulates all items sharing high-level semantic features represented by the codebook embedding e 𝑙,𝑗 .

At this level, each semantic group G 𝑙,𝑗 serves as a topological unit. The teacher representation of G 𝑙,𝑗 is defined as the group-level semantic centroid in the continuous embedding space:

$$h _ { j , I G } ^ { t } = \frac { 1 } { | \mathcal { G } _ { l , j } | } \sum _ { i \in \mathcal { G } _ { l , j } } s _ { i } ,$$

which summarizes the shared semantics of items within the group. The student representation of each semantic group is defined as the corresponding code embedding h s 𝑗, IG = e 𝑙,𝑗 , serving as the grouplevel prototype in the tokenized space.

To model the topological structure between item groups, we construct the pairwise inter-group distance matrices D t IG and D s IG in the teacher and student spaces, where each entry is defined as:

$$d _ { j k } ^ { t } = \| h _ { j , I G } ^ { t } - h _ { k , I G } ^ { t } \| ^ { 2 } , \quad d _ { j k } ^ { s } = \| h _ { j , I G } ^ { s } - h _ { k , I G } ^ { s } \| ^ { 2 } , \quad \ \ ( 1 2 )$$

which encode the global relative arrangement of semantic groups.

By instantiating the topology distillation objective (Eqs. 8 and 9) on inter-group distance matrices, this level enforces consistency in neighborhood rankings among coarse-grained item groups across the embedding and tokenized spaces, thereby anchoring global group-level structure and preventing early-stage topological errors from propagating through the quantization hierarchy.

Intra-Group (IaG) . While inter-group distillation stabilizes coarse semantic organization, it leaves the internal structure of each group unconstrained. Intra-group distillation bridges this gap by refining local topology at an intermediate semantic granularity, where tokenized representations are enforced to capture fine-grained relationships among items within each group.

At this level, each item 𝑖 within semantic groups serves as a topological unit. At the residual layer 𝑙 applied intra-group distillation, the teacher representation of item 𝑖 is defined as the original item embedding, and the student representation is the cumulative reconstructed representation up to the quantization depth 𝑙 :

$$h _ { i , I a G } ^ { t } = s _ { i } , \ \ h _ { i , I a G } ^ { s } = \sum _ { m = 1 } ^ { l } e _ { m , c _ { i , m } } .$$

To focus topology supervision on local structure, we restrict distance computation to item pairs belonging to the same semantic group determined at the preceding layer 𝑙 - 1 (Eq. 3). The teacher and student intra-group distance matrices are defined as:

$$d _ { i j } ^ { t } = \begin{cases} \| h _ { i , I a G } ^ { t } - h _ { j , I a G } ^ { t } \| ^ { 2 } , & \text {if } c _ { i , l - 1 } = c _ { j , l - 1 } \text { and } i \neq j , \\ \infty , & \text {otherwise} , \end{cases} \quad ( 1 4 )$$

$$d _ { i j } ^ { s } = \begin{cases} \| h _ { i , I a G } ^ { s } - h _ { j , I a G } ^ { s } \| ^ { 2 } , & \text {if } c _ { i , l - 1 } = c _ { j , l - 1 } \text { and } i \neq j , \\ \infty , & \text {otherwise} , \end{cases} ( 1 5 )$$

Here, distances for item pairs from different groups are set to ∞ , ensuring that the topology supervision in Eq. 8 focuses exclusively on intra-group neighborhood structure without affecting global group-level structure. By integrating intra-group distances into the foundational template (Eqs. 8 and 9), intra-group distillation reinforces neighborhood rankings within each semantic group. This level is crucial for recovering fine-grained local structure that is often attenuated during item tokenization, thereby enhancing the model's ability to discriminate between closely related items.

Inter-Item (II) . At deeper stages of RQ-VAE, we introduce interitem distillation to preserve the global neighborhood structure at the fine-grained item level. Each item 𝑖 serves as a topological unit. The teacher representation is the original semantic embedding, while the student representation corresponds to the reconstructed representation aggregated up to the given quantization depth 𝑙 :

$$h _ { i , \mathbb { I } } ^ { t } = s _ { i } , \ \ h _ { i , \mathbb { I } } ^ { s } = \sum _ { m = 1 } ^ { l } e _ { m , c _ { i , m } } , \quad ( 1 6 )$$

where 𝑙 denotes the layer depth at which item-level topology supervision is applied. To capture the global item-level topological structure, we construct pairwise inter-item distance matrices:

$$d _ { i j } ^ { t } = \| \mathbf h _ { _ { i , \mathbb { I } } } ^ { t } - \mathbf h _ { _ { j , \mathbb { I } } } ^ { t } \| ^ { 2 } , \ \ d _ { i j } ^ { s } = \| \mathbf h _ { _ { i , \mathbb { I } } } ^ { s } - \mathbf h _ { _ { j , \mathbb { I } } } ^ { s } \| ^ { 2 } .$$

These matrices encode the comprehensive relational structure among items in the embedding and tokenized spaces.

By substituting inter-item distance matrices into Eqs. 8 and 9, inter-item distillation enforces global neighborhood ranking consistency at the finest semantic granularity.

3.2.3 Hierarchical Layer Deployment . This section details how distillation at different granularities is assigned across the quantization layers to align with the hierarchical structure of RQ-VAE.

Let 𝑙 ∈ { 1 , . . . , 𝐿 } denote the index of a residual quantization layer in an 𝐿 -layer RQ-VAE. We map the three topology distillation levels according to the semantic granularity captured at each layer:

- Inter-group distillation is anchored at the first residual layer ( 𝑙 = 1), where items are encoded into coarse semantic clusters. This distillation level establishes a global structure backbone by aligning neighborhood relations among high-level item groups.
- Intra-group distillation is applied at the second layer ( 𝑙 = 2), where representations begin to differentiate within established groups. Topology supervision at this stage refines local manifold structure while preserving group-level boundaries.
- Inter-item distillation is applied at deeper layers (2 &lt; 𝑙 ≤ 𝐿 ), where residual codes encode fine-grained semantic variations. At this stage, topology supervision operates directly at the item level to preserve global point-to-point neighborhood consistency, completing the coarse-to-fine alignment process.

In special configurations such as a two-layer RQ-VAE ( 𝐿 = 2), we propose an inter-group and inter-item combination. This configuration preserves both global structure and item-level discriminability while respecting the hierarchical nature of residual quantization.

Beyond this fixed assignment, we further recommend using the codebook utilization rate as a practical indicator of semantic granularity to flexibly determine the optimal distillation level for each layer. Low-utilization layers, where items concentrate on a small subset of codes, encode coarse semantics and are well-suited for inter-group distillation. Moderately utilized layers capture intermediate semantics and benefit from intra-group distillation. Highly utilized layers, where most or all codes are activated, encode fine residual structure and are best supervised by inter-item distillation. Typically, utilization rates in RQ-VAE tokenizers increase with depth, supporting our progressive deployment.

Overall, this stage-based deployment makes TopoTok architectureagnostic and readily applicable to RQ-VAE frameworks of arbitrary depth. By aligning distillation granularity with the quantization hierarchy, TopoTok provides a general solution for generative recommendation models built on learnable hierarchical tokenization, achieving structural flexibility without architectural changes.

3.2.4 Training Objective . The proposed TopoTok is integrated into RQ-VAE backbone to guide the training of a topology-aware semantic ID tokenizer. The overall training objective is defined as:

$$\mathcal { L } _ { t o t a l } = \mathcal { L } _ { R Q \text {-VAE} } + \alpha \cdot \mathcal { L } _ { T o p o T o k } ,$$

where 𝛼 is the topology distillation weight that controls the strength of topology supervision. The TopoTok distillation loss is given by:

$$\mathcal { L } _ { \text {Topok} } = \mathcal { L } _ { \text {inter-group} } + \mathcal { L } _ { \text {intra-group} } + \mathcal { L } _ { \text {inter-item} } ,$$

This joint training objective enables RQ-VAE to preserve both item semantic information and the hierarchical topology information among items, resulting in higher-quality tokenization that better supports generative recommendation. For training complexity, TopoTok relies solely on batch-local, highly parallelizable computations, incurring a modest training overhead. Importantly, TopoTok introduces no additional computation at inference time , as the distillation objectives are only applied during training and do not alter the tokenization or recommendation pipeline at serving time. This property is particularly desirable in recommender systems, where inference latency is critical for real-time deployment.

## 4 EXPERIMENTS

## 4.1 Experimental Setting

- 4.1.1 Dataset. We evaluate our method following the standard protocol used in prior work [28, 33]. Experiments are conducted on three subsets of the latest Amazon Review dataset [10]: Industrial Scientific , Musical Instruments , and Video Games . We apply a 5-core filtering procedure, removing users and items with fewer than five interactions. User interaction sequences are then constructed in chronological order, with a maximum sequence length of 20.
- 4.1.2 Baseline Models. We compare TopoTok with comprehensive baselines from three categories: (1) Traditional sequential recommendation methods: Caser [31], GRU4Rec [14], SASRec [17], BERT4Rec [29], FDSA [38], and S 3 Rec [41]. (2) Generative recommendation methods: P5-CID [13], P5-SID [13], TIGER [28], and ETEGRec [22]. (3) Semantic ID tokenization enhancement methods: LETTER [33] and CoST [42] (both implemented on TIGER).

4.1.3 Evaluation Protocol and Implementation Details. We evaluate all models using top-K Recall (R@K) and NDCG (N@K) with 𝐾 = { 5 , 10 } . Following standard practice [28], we adopt the leaveone-out strategy: for each user, the last interaction is used for testing, the second-last for validation, and the rest for training. We conduct a full-ranking evaluation over the entire candidate item set without sampling. We adopt TIGER [28] and ETEGRec [22] as backbone generative recommenders, using Sentence-T5 [26] and SASRec [17] to obtain item embeddings, respectively. For item tokenization, we use RQ-VAE with three codebook layers (each with 256 codes of dimension 128, except for Section 4.3.2). TopoTok is trained for 10k epochs using AdamW [24] (lr=1e-3, batch size=2048), with the topology weight 𝛼 ∈ { 0 . 01 , 0 . 1 , 0 . 3 , 0 . 5 , 1 } selected on validation. Following TIGER [28], we append an additional token to ensure semantic ID uniqueness. We use T5 as our recommender and follow the original training protocols [22, 28]. All experiments are conducted on a single NVIDIA Tesla A40 GPU. Results are reported using the model with the best validation NDCG@10. Statistical significance is assessed via a paired t-test over five independent runs.

Table 1: Performance comparison across three datasets. The best and second-best results within each comparison group are highlighted in bold and underlined font, respectively. Superscript ∗ indicates statistical significance at 𝑝 &lt; 0 . 05 .

| Model           |          |                 |          |          | Instrument   | Instrument   | Instrument   | Instrument   | Game     | Game     | Game     | Game     |
|-----------------|----------|-----------------|----------|----------|--------------|--------------|--------------|--------------|----------|----------|----------|----------|
| Model           | R@5      | Scientific R@10 | N@5      | N@10     | R@5          | R@10         | N@5          | N@10         | R@5      | R@10     | N@5      | N@10     |
| Caser           | 0.0172   | 0.0281          | 0.0107   | 0.0142   | 0.0242       | 0.0392       | 0.0154       | 0.0202       | 0.0346   | 0.0567   | 0.0221   | 0.0291   |
| GRU4Rec         | 0.0221   | 0.0353          | 0.0144   | 0.0186   | 0.0345       | 0.0537       | 0.0220       | 0.0281       | 0.0522   | 0.0831   | 0.0337   | 0.0436   |
| SASRec          | 0.0256   | 0.0406          | 0.0147   | 0.0195   | 0.0341       | 0.0530       | 0.0217       | 0.0277       | 0.0517   | 0.0821   | 0.0329   | 0.0426   |
| BERT4Rec        | 0.0180   | 0.0300          | 0.0113   | 0.0151   | 0.0305       | 0.0483       | 0.0196       | 0.0253       | 0.0453   | 0.0716   | 0.0294   | 0.0378   |
| FDSA            | 0.0261   | 0.0391          | 0.0174   | 0.0216   | 0.0364       | 0.0557       | 0.0233       | 0.0295       | 0.0548   | 0.0857   | 0.0353   | 0.0453   |
| S 3 Rec         | 0.0253   | 0.0410          | 0.0172   | 0.0218   | 0.0340       | 0.0538       | 0.0218       | 0.0282       | 0.0533   | 0.0823   | 0.0351   | 0.0444   |
| P5-SID          | 0.0155   | 0.0234          | 0.0103   | 0.0129   | 0.0319       | 0.0438       | 0.0237       | 0.0275       | 0.0480   | 0.0693   | 0.0333   | 0.0401   |
| P5-CID          | 0.0192   | 0.0300          | 0.0123   | 0.0158   | 0.0352       | 0.0507       | 0.0234       | 0.0285       | 0.0497   | 0.0748   | 0.0343   | 0.0424   |
| TIGER           | 0.0275   | 0.0431          | 0.0181   | 0.0231   | 0.0368       | 0.0574       | 0.0242       | 0.0308       | 0.0570   | 0.0895   | 0.0370   | 0.0471   |
| LETTER          | 0.0276   | 0.0433          | 0.0179   | 0.0230   | 0.0372       | 0.0581       | 0.0243       | 0.0310       | 0.0576   | 0.0901   | 0.0373   | 0.0475   |
| CoST            | 0.0270   | 0.0426          | 0.0180   | 0.0229   | 0.0366       | 0.0570       | 0.0242       | 0.0306       | 0.0569   | 0.0897   | 0.0379   | 0.0472   |
| TIGER-TopoTok   | 0.0302 * | 0.0465 *        | 0.0196 * | 0.0248 * | 0.0402 *     | 0.0613 *     | 0.0263 *     | 0.0331 *     | 0.0608 * | 0.0939 * | 0.0403 * | 0.0509 * |
| Improv.         | +9.42%   | +7.39%          | +8.29%   | +7.36%   | +8.06%       | +5.51%       | +8.23%       | +6.77%       | +5.56%   | +4.22%   | +6.33%   | +7.84%   |
| ETEGRec         | 0.0294   | 0.0455          | 0.0190   | 0.0241   | 0.0402       | 0.0624       | 0.0260       | 0.0331       | 0.0616   | 0.0947   | 0.0400   | 0.0507   |
| ETEGRec-TopoTok | 0.0307*  | 0.0481*         | 0.0201*  | 0.0255*  | 0.0426*      | 0.0657*      | 0.0273*      | 0.0349*      | 0.0635*  | 0.0975*  | 0.0414*  | 0.0525*  |
| Improv.         | +4.42%   | +5.71%          | +5.79%   | +5.81%   | +5.97%       | +5.29%       | +5.00%       | +5.44%       | +3.08%   | +2.96%   | +3.50%   | +3.55%   |

For the main results in Table 1, we report performance using seed 2025 for reproducibility. Unless otherwise specified, experiments are conducted on the TIGER backbone.

## 4.2 Overall Performance

We evaluate TopoTok under two representative generative recommendation paradigms: TIGER [28], which trains the tokenizer and recommender sequentially, and ETEGRec [22], which adopts end-to-end training. Following their original designs, tokenization enhancement methods such as LETTER [33] and CoST [42] are implemented under the TIGER backbone and included in the TIGER-based comparison. The overall results are in Table 1.

- TopoTok consistently improves generative recommendation across both backbones. Under the TIGER backbone, TIGERTopoTok achieves the best performance across all datasets and metrics, delivering statistically significant gains over TIGER and prior tokenization enhancement methods such as LETTER and CoST (e.g., up to + 9 . 42% on Scientific). When integrated into the end-to-end ETEGRec framework, ETEGRec-TopoTok further establishes new state-of-the-art results. These results demonstrate that TopoTok is robust in both two-stage and end-to-end settings.
- End-to-end training with TopoTok unlocks the full potential of RQ-VAE tokenization. ETEGRec-TopoTok consistently surpasses TIGER-TopoTok across all datasets, demonstrating the benefit of jointly optimizing item tokenization and generation. This advantage stems from the parameterized nature of RQ-VAE, which allows gradients from the generative objective to propagate back to the tokenizer. By providing topology-aware supervision throughout the quantization hierarchy, TopoTok stabilizes end-to-end optimization and enables RQ-VAE to preserve relational structure while adapting token representations to downstream generation.
- Topology supervision must respect the hierarchical structure of residual quantization. Within the TIGER-based group, although LETTER and CoST provide improvements, their gains are

limited by a lack of hierarchical awareness. Specifically, LETTER does not incorporate explicit topology supervision, while CoST applies monolithic contrastive supervision only at the final output, overlooking the layer-wise semantic progression of RQ-VAE. We further observe that CoST exhibits inconsistent performance across datasets, which we attribute to its limited ability to preserve topology under hierarchical semantics, as supported by the analysis in Section 4.5. Moreover, applying contrastive supervision only at the final layer may interfere with reconstruction learning, leading to degraded performance in Table 1. In contrast, TopoTok explicitly aligns topology supervision with the coarse-to-fine structure of RQ-VAE, enabling each codebook layer to preserve topology at an appropriate semantic scale.

## 4.3 Ablation Study

- 4.3.1 Multi-Granularity Topology Distillation. To evaluate the effectiveness of our hierarchical topology distillation, we conduct ablation studies by incrementally adding Inter-Group (IG), IntraGroup (IaG), and Inter-Item (II) Distillation on the TIGER backbone. We also report the top-20 neighborhood overlap across the three codebook layers, which is computed as the average ratio of shared top-20 nearest neighbors between the original semantic embeddings and the reconstructed representations at each layer. Higher overlap indicates better topology preservation during tokenization. From the results in Table 2, we derive the following conclusions:
- The three levels of topology distillation are complementary and mutually reinforcing. For example, introducing IG alone leads to notable gains in overlap across all layers, suggesting that coarse-grained topology supervision at the early layer induces structural adjustments that mitigate the accumulation of topology distortion in deeper layers. Conversely, adding supervision at deeper layers also leads to improvements in earlier layers.
- TopoTok alleviates topology distortion. The full TopoTok achieves the highest neighborhood overlaps across all datasets,

Table 2: Ablation study of TopoTok components. IG, IaG, and II denote Inter-Group, Intra-Group, and Inter-Item distillation, respectively. l indicates the codebook layer.

| Variants     | Variants   | Variants   | Metrics   | Metrics   | Metrics   | Metrics   | Top-20 Overlap   | Top-20 Overlap   | Top-20 Overlap   |
|--------------|------------|------------|-----------|-----------|-----------|-----------|------------------|------------------|------------------|
| IG           | IaG        | II         | R@5       | R@10      | N@5       | N@10      | l1               | l2               | l3               |
| -            | -          | -          | 0.0275    | 0.0431    | 0.0181    | 0.0231    | 62.90%           | 30.29%           | 27.55%           |
| ✓            | -          | -          | 0.0288    | 0.0450    | 0.0187    | 0.0239    | 72.47%           | 35.99%           | 31.43%           |
| Scientific - | ✓          | -          | 0.0280    | 0.0439    | 0.0183    | 0.0234    | 66.12%           | 32.88%           | 32.05%           |
| -            | -          | ✓          | 0.0285    | 0.0443    | 0.0189    | 0.0240    | 66.29%           | 31.54%           | 32.94%           |
| ✓            | ✓          | -          | 0.0293    | 0.0461    | 0.0193    | 0.0243    | 73.46%           | 39.05%           | 32.56%           |
| ✓            | ✓          | ✓          | 0.0302    | 0.0465    | 0.0196    | 0.0248    | 73.70%           | 41.26%           | 36.75%           |
| -            | -          | -          | 0.0368    | 0.0574    | 0.0242    | 0.0308    | 63.06%           | 27.62%           | 26.49%           |
| ✓            | -          | -          | 0.0386    | 0.0590    | 0.0259    | 0.0324    | 65.69%           | 31.13%           | 27.22%           |
| Instrument - | ✓          | -          | 0.0391    | 0.0606    | 0.0259    | 0.0328    | 64.27%           | 29.86%           | 31.71%           |
| -            | -          | ✓          | 0.0382    | 0.0586    | 0.0252    | 0.0318    | 63.06%           | 27.62%           | 30.09%           |
| ✓            | ✓          | -          | 0.0397    | 0.0609    | 0.0263    | 0.0330    | 72.07%           | 33.12%           | 28.78%           |
| ✓            | ✓          | ✓          | 0.0402    | 0.0613    | 0.0263    | 0.0331    | 74.05%           | 34.60%           | 29.20%           |
| -            | -          | -          | 0.0570    | 0.0895    | 0.0370    | 0.0471    | 55.08%           | 27.97%           | 27.57%           |
| ✓            | -          | -          | 0.0587    | 0.0912    | 0.0387    | 0.0491    | 57.12%           | 30.49%           | 29.92%           |
| Game -       | ✓          | -          | 0.0588    | 0.0914    | 0.0388    | 0.0492    | 58.27%           | 32.45%           | 27.57%           |
| -            | -          | ✓          | 0.0582    | 0.0908    | 0.0381    | 0.0485    | 55.08%           | 27.75%           | 29.29%           |
| ✓            | ✓          | -          | 0.0595    | 0.0931    | 0.0391    | 0.0500    | 61.61%           | 31.90%           | 30.74%           |
| ✓            | ✓          | ✓          | 0.0608    | 0.0939    | 0.0403    | 0.0509    | 63.30%           | 35.67%           | 33.36%           |

Table 3: Robustness of TopoTok across different RQ-VAE layers (L). Bold indicates the better result for each configuration.

| Dataset    | Layers (L)   | Model         | Recall@10     | NDCG@10       | Improv.   |
|------------|--------------|---------------|---------------|---------------|-----------|
| Scientific | L=2          | Base +TopoTok | 0.0400 0.0445 | 0.0213 0.0237 | +11.25%   |
|            | L=3          | Base +TopoTok | 0.0431 0.0465 | 0.0231 0.0248 | +7.89%    |
|            | L=4          | Base +TopoTok | 0.0457 0.0468 | 0.0246 0.0249 | +2.41%    |
|            | L=2          | Base +TopoTok | 0.0569 0.0595 | 0.0303 0.0322 | +4.57%    |
| Instrument | L=3          | Base +TopoTok | 0.0574 0.0613 | 0.0308 0.0331 | +6.79%    |
|            | L=4          | Base +TopoTok | 0.0601 0.0617 | 0.0329 0.0336 | +2.66%    |

confirming its ability to preserve multi-level topological structure. The corresponding improvements in recommendation performance further support the conclusion that reducing topology distortion enhances token quality and benefits downstream generation.

4.3.2 Quantization Depth. We conduct an ablation study to examine the robustness of TopoTok under different RQ-VAE quantization depths. Experiments are performed on TIGER backbone, with the number of residual layers set to the commonly adopted configurations 𝐿 ∈ { 2 , 3 , 4 } . The layer-wise deployment of topology distillation follows the strategy described in Section 3.2.3. Performance improvements are computed based on Recall@10.

As shown in Table 3, TopoTok consistently improves recommendation performance across all tested quantization depths and datasets, demonstrating stability across layer configurations. The relative gains are most pronounced when 𝐿 = 2, where the shallow RQ-VAE hierarchy leads to greater information loss during residual quantization. In this setting, topology supervision provided by TopoTok effectively compensates for the limited representational capacity. As the number of residual layers increases ( 𝐿 = 3 , 4), the tokenizer captures progressively finer semantic structure, and TopoTok continues to deliver stable improvements, confirming its compatibility with deeper hierarchical tokenization. These results show that by aligning topology supervision with the semantic roles of residual layers, TopoTok remains effective across varying RQVAE depths, highlighting its generality and practical applicability.

Figure 3: Performance across different 𝛼 on three datasets.

<!-- image -->

## 4.4 Hyperparameter Analysis

We conduct controlled experiments on three datasets to examine the effect of the topology distillation weight 𝛼 , with results shown in Figure 3. As 𝛼 increases from small values, both Recall@10 and NDCG@10 consistently improve across datasets, indicating that introducing topology supervision effectively guides the tokenizer to preserve relational structure. Performance peaks at moderate values of 𝛼 , after which further increasing 𝛼 leads to a clear decline. This trend suggests that overly strong topology constraints introduce excessive regularization, which hampers semantic reconstruction during vector quantization. The optimal 𝛼 is datasetdependent: 𝛼 = 0 . 1 yields the best performance on the Scientific and Instrument datasets, while 𝛼 = 0 . 3 is optimal for the Game dataset. Overall, these results highlight the importance of balancing topology supervision with reconstruction objectives to achieve optimal tokenization quality.

## 4.5 Visualization Case Study

While each distillation objective of TopoTok improves topology preservation (as shown in Table 2), we further qualitatively assess the impact of topology-aware distillation comparing TIGER, CoST, and TIGER-TopoTok. Specifically, we conduct a visualization case study comparing TIGER, CoST, and TIGER-TopoTok. The goal is to examine how well the neighborhood structure in the semantic space is preserved across quantization layers.

· Experimental Setup. We randomly select a query item from the Instrument dataset and retrieve its top-20 nearest neighbors in the semantic space and the tokenized space at three quantization layers, forming a subset of sampled items for analysis. The resulting rank pairs are visualized in scatter plots, where the 𝑥 -axis denotes the rank in the semantic space, and the 𝑦 -axis denotes the rank in the tokenized space. The diagonal line represents perfect alignment between the two rankings, reflecting ideal topological preservation.

The case study consists of two parts. First, we highlight the top-20 semantic neighbors of the query item in color, enabling a clear inspection of their rank consistency across layers. Second, to mitigate the randomness of the query item, we plot all rank pairs of other sampled items in gray, offering a broader view of general trends in topological preservation.

· Interpretation Criteria. A desirable result exhibits two key patterns. First, most colored points fall within or near the bottom-left highlighted region (rank ≤ 20 in both spaces), indicating minimal distortion in the query item's top-20 neighborhood. Second, gray points concentrated along or near the diagonal, reflecting generally consistent relative rankings between the semantic and reconstructed spaces. Together, these patterns suggest that the semantic topology is well preserved during item tokenization.

Figure 4: Rank comparison between semantic and reconstructed spaces at each layer in TIGER.

<!-- image -->

Figure 5: Rank comparison between semantic and reconstructed spaces at each layer in CoST.

<!-- image -->

Figure 6: Rank comparison between semantic and reconstructed spaces at each layer in TopoTok.

<!-- image -->

· Results and Comparison. Figures 4, 5, and 6 present rank-scatter plots for TIGER, CoST, and TopoTok on the same query item, B09V188Y4X. The semantic IDs of the query item and its neighbors are reported in Tables 4 for reference.

TIGER preserves rank alignment at layer 1, where most top-20 semantic neighbors share the same first semantic ID. However, this alignment degrades in deeper layers, with many neighbors falling outside the top-20 range and points increasingly deviating from the diagonal, indicating topology distortion. CoST exhibits similar behavior: while some local structure is retained at layer 1, rank consistency deteriorates in layers 2 and 3. This behavior stems from applying a monolithic contrastive objective, without respecting the hierarchical semantics encoded across residual layers

In contrast, TopoTok consistently preserves topological structure across all layers. In layers 1 and 2, most of the top-20 semantic neighbors of the query item are ranked first, indicating that they share the same first two semantic IDs, 50 and 207. In layer 3, more neighbors remain within the top-20 reconstructed neighborhood, with points more tightly concentrated along the diagonal. Furthermore, the gray points are distributed closer to the diagonal, suggesting that the relational structure is more accurately preserved. These results demonstrate that explicitly aligning topology supervision with the coarse-to-fine quantization hierarchy enables TopoTok to achieve superior multi-level topology preservation.

Table 4: Semantic IDs of the query item and its top-20 semantic neighbors across TIGER, CoST, and TopoTok. Neighbors are sorted by ascending semantic-space rank (1-20).

| Rank   | Item asin   | IDs (TIGER)     | IDs (CoST)      | IDs (TopoTok)   |
|--------|-------------|-----------------|-----------------|-----------------|
| Query  | B09V188Y4X  | [106, 56, 21]   | [116, 193, 145] | [50, 207, 136]  |
| 1      | B093L2LHB9  | [106, 220, 0]   | [116, 193, 145] | [50, 207, 136]  |
| 2      | B089GTNJYQ  | [106, 56, 0]    | [116, 193, 145] | [50, 207, 36]   |
| 3      | B08372HW3L  | [106, 51, 222]  | [116, 65, 239]  | [50, 207, 76]   |
| 4      | B083ZFH24H  | [106, 220, 0]   | [116, 65, 31]   | [50, 207, 136]  |
| 5      | B085KW8K3F  | [106, 56, 21]   | [116, 193, 145] | [50, 207, 136]  |
| 6      | B085XF93S7  | [106, 30, 21]   | [116, 233, 142] | [50, 207, 245]  |
| 7      | B085DM132N  | [106, 56, 0]    | [116, 193, 145] | [50, 207, 17]   |
| 8      | B08PK7CDKW  | [106, 56, 21]   | [116, 65, 31]   | [50, 207, 136]  |
| 9      | B08H7Y1HQY  | [106, 56, 21]   | [116, 193, 145] | [50, 207, 136]  |
| 10     | B082KZ3R2F  | [106, 51, 222]  | [252, 65, 37]   | [50, 207, 36]   |
| 11     | B0B5QTG996  | [106, 192, 187] | [116, 57, 145]  | [44, 207, 50]   |
| 12     | B07CZD8S8H  | [106, 192, 7]   | [116, 193, 235] | [50, 207, 115]  |
| 13     | B07X5VT56K  | [106, 220, 161] | [116, 65, 34]   | [50, 207, 109]  |
| 14     | B0823DMFG2  | [106, 51, 222]  | [116, 65, 34]   | [50, 207, 115]  |
| 15     | B08P7CMR1Q  | [106, 56, 21]   | [116, 193, 145] | [50, 207, 191]  |
| 16     | B0B1DJ7BB7  | [106, 220, 49]  | [116, 65, 145]  | [50, 207, 24]   |
| 17     | B07VT2YD88  | [106, 220, 210] | [116, 193, 145] | [50, 207, 109]  |
| 18     | B0B1DN5CSJ  | [106, 220, 233] | [116, 65, 145]  | [50, 207, 24]   |
| 19     | B0BKFZP9KR  | [106, 220, 49]  | [116, 65, 145]  | [50, 207, 190]  |
| 20     | B09S9SMDZK  | [106, 220, 25]  | [116, 73, 69]   | [50, 207, 115]  |

## 5 CONCLUSION

In this paper, we identify the underexplored topology distortion problem in existing semantic ID-based item tokenization for generative recommendation. To address this issue, we propose a novel Topo logy-Aware Tok enization framework (TopoTok), which decomposes topology supervision into three coarse-to-fine levels, aligning with the residual quantization hierarchy. Extensive experiments demonstrate that TopoTok significantly enhances topology preservation and outperforms state-of-the-art baselines. TopoTok offers a general solution for preserving topological structure in item tokenization across various generative recommendation models.

## Acknowledgments

This research is supported in part by the National Science Foundation under Grant No. CNS-2427070, IIS-2331069, IIS-2202481, IIS2130263, CNS-2131622. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation hereon.

## References

- [1] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yanchen Luo, Chong Chen, Fuli Feng, and Qi Tian. 2025. A bi-step grounding paradigm for large language models in recommendation systems. ACM Transactions on Recommender Systems 3, 4 (2025), 1-27.
- [2] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems . 1007-1014.
- [3] Zhixuan Chu, Hongyan Hao, Xin Ouyang, Simeng Wang, Yan Wang, Yue Shen, Jinjie Gu, Qing Cui, Longfei Li, Siqiao Xue, et al. 2023. Leveraging large language models for pre-trained recommender systems. arXiv preprint arXiv:2308.10837 (2023).
- [4] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems . 191-198.
- [5] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering chatgpt's capabilities in recommender systems. In Proceedings of the 17th ACM Conference on Recommender Systems . 1126-1132.
- [6] Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys). In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Barcelona, Spain) (KDD '24) . Association for Computing Machinery, New York, NY, USA, 6448-6458. doi:10.1145/3637528. 3671474
- [7] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965 (2025).
- [8] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence 36, 4 (2013), 744-755.
- [9] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt &amp; predict paradigm (p5). In Proceedings of the 16th ACM conference on recommender systems . 299-315.
- [10] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024. Bridging Language and Items for Retrieval and Recommendation. arXiv preprint arXiv:2403.03952 (2024).
- [11] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large language models are zero-shot rankers for recommender systems. In European Conference on Information Retrieval . Springer, 364-381.
- [12] Michael E Houle and Michael Nett. 2014. Rank-based similarity search: Reducing the dimensional dependence. IEEE transactions on pattern analysis and machine intelligence 37, 1 (2014), 136-150.
- [13] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to index item ids for recommendation foundation models. In Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region . 195-204.
- [14] Dietmar Jannach and Malte Ludewig. 2017. When recurrent neural networks meet the neighborhood for session-based recommendation. In Proceedings of the eleventh ACM conference on recommender systems . 306-310.
- [15] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117-128.
- [16] SeongKu Kang, Junyoung Hwang, Wonbin Kweon, and Hwanjo Yu. 2021. Topology distillation for recommender system. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining . 829-839.
- [17] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM) . IEEE, 197-206.
- [18] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 11523-11532.
- [19] Yongqi Li, Nan Yang, Liang Wang, Furu Wei, and Wenjie Li. 2023. Generative retrieval for conversational question answering. Information Processing &amp; Management 60, 5 (2023), 103475.
- [20] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, and Xiang Wang. 2023. Llara: Aligning large language models with sequential recommenders. CoRR (2023).
- [21] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan He. 2024. Llara: Large language-recommendation assistant. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1785-1795.
- [22] Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, and Wayne Xin Zhao. 2024. Generative Recommender with End-to-End Learnable Item Tokenization. arXiv preprint arXiv:2409.05546 (2024).
- [23] Yifan Liu, Yaokun Liu, Zelin Li, Zhenrui Yue, Gyuseok Lee, Ruichen Yao, Yang Zhang, and Dong Wang. 2025. Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation. arXiv preprint arXiv:2509.10468 (2025).
- [24] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019 .
- [25] Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. IEEE transactions on pattern analysis and machine intelligence 36, 11 (2014), 2227-2240.
- [26] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-t5: Scalable sentence encoders from pretrained text-to-text models. In Findings of the association for computational linguistics: ACL 2022 . 1864-1874.
- [27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1-67.
- [28] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299-10315.
- [29] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management . 1441-1450.
- [30] Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, and Yongfeng Zhang. 2024. Towards llm-recsys alignment with textual id learning. arXiv e-prints (2024), arXiv-2403.
- [31] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining . 565-573.
- [32] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
- [33] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, SeeKiong Ng, and Tat-Seng Chua. 2024. Learnable Item Tokenization for Generative Recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (Boise, ID, USA) (CIKM '24) . Association for Computing Machinery, New York, NY, USA, 2400-2409. doi:10.1145/3627673. 3679569
- [34] Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, et al. 2024. Enhanced generative recommendation via content and collaboration integration. CoRR (2024).
- [35] Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024. Eager: Two-stream generative recommender with behavior-semantic collaboration. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 3245-3254.
- [36] Jun Yin, Zhengxin Zeng, Mingzheng Li, Hao Yan, Chaozhuo Li, Weihao Han, Jianjin Zhang, Ruochen Liu, Hao Sun, Weiwei Deng, et al. 2025. Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism. In Proceedings of the ACM on Web Conference 2025 . 216-227.
- [37] Junjie Zhang, Ruobing Xie, Yupeng Hou, Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2025. Recommendation as instruction following: A large language model empowered recommendation approach. ACM Transactions on Information Systems 43, 5 (2025), 1-37.
- [38] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level deeper selfattention network for sequential recommendation.. In IJCAI . 4320-4326.
- [39] Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. (2021).
- [40] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE) . IEEE, 1435-1448.
- [41] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM international conference on information &amp; knowledge management . 1893-1902.
- [42] Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, and Xiu Li. 2024. Cost: Contrastive quantization based semantic tokenization for generative recommendation. In Proceedings of the 18th ACM Conference on Recommender

RecSys '26, September 27-October 02, 2026, Minneapolis, MN, USA Yaokun Liu et al.

Systems . 969-974. ACM Web Conference 2024 . 3162-3172.

- [43] Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, and Jundong Li. 2024. Collaborative large language model for recommender systems. In Proceedings of the