## Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation

Meng Yuan

Institute of Artificial Intelligence, Beihang University Beijing, China yuanmeng97@buaa.edu.cn Fuzhen Zhuang ∗†

Institute of Artificial Intelligence, Beihang University Beijing, China zhuangfuzhen@buaa.edu.cn

## Deqing Wang

School of Computer Science and Engineering, Beihang University Beijing, China dqwang@buaa.edu.cn Zhao Zhang Institute of Computing Technology, Chinese Academy of Sciences Beijing, China zhangzhao2021@ict.ac.cn Jin Dong Beijing Academy of Blockchain and Edge Computing Beijing, China dongjin@baec.org.cn

## CCS CONCEPTS

· Information systems → Recommender systems .

## KEYWORDS

Knowledge Graph, Recommender Systems, Multiple Space Fusion, Geometry-aware Optimization Strategy

## ACMReference Format:

Meng Yuan, Fuzhen Zhuang, Zhao Zhang, Deqing Wang, and Jin Dong. 2023. Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation. In Seventeenth ACM Conference on Recommender Systems (RecSys '23), September 18-22, 2023, Singapore, Singapore. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3604915.3608787

## 1 INTRODUCTION

Recommender systems (RSs) have shown great potential in solving the information explosion problem and enhancing the user experience in various online applications [27]. In a variety of scenarios, knowledge graphs (KGs) can be utilized to offer fundamental background knowledge as well as rich structural information. To fully utilize KG, existing knowledge-enhanced recommendation methods [4, 16, 22, 30, 36] strive to build more effective neural networks to integrate the semantics of KG.

Although these techniques can significantly enhance the embeddings of users and items, they are entirely designed based on Euclidean space without considering curvature. Recent studies [1, 21, 39] have revealed that a tremendous data exhibits the highly non-Euclidean properties, especially the data that presents tree-likeness or cyclic structures. For instance, the KG data are diverse and include a variety of structures. As shown in Figure 1, there are interactions between the user and items, which can be further connected to more entities. Owing to such high-order relationship [34], knowledge-aware methods have the ability to provide accurate recommendations. Essentially, such interactions can be seen as tree structures or cyclic structures. In such situations, Euclidean space methods suffer from severe distortion when depicting these structures [3, 23]. Furthermore, recent studies [10, 18] have demonstrated that geometric spaces with constant non-zero curvature can enhance representation performance when the underlying graph structures of the data follow particular patterns. Specifically, the hyperbolic space is more suitable for modeling tree-structured

## ABSTRACT

Since Knowledge Graphs (KGs) contain rich semantic information, recently there has been an influx of KG-enhanced recommendation methods. Most of existing methods are entirely designed based on euclidean space without considering curvature. However, recent studies have revealed that a tremendous graph-structured data exhibits highly non-euclidean properties. Motivated by these observations, in this work, we propose a knowledge-based multiple adaptive spaces fusion method for recommendation, namely MCKG. Unlike existing methods that solely adopt a specific manifold, we introduce the unified space that is compatible with hyperbolic, euclidean and spherical spaces. Furthermore, we fuse the multiple unified spaces in an attention manner to obtain the high-quality embeddings for better knowledge propagation. In addition, we propose a geometry-aware optimization strategy which enables the pull and push processes benefited from both hyperbolic and spherical spaces. Specifically, in hyperbolic space, we set smaller margins in the area near to the origin, which is conducive to distinguishing between highly similar positive items and negative ones. At the same time, we set larger margins in the area far from the origin to ensure the model has sufficient error tolerance. The similar manner also applies to spherical spaces. Extensive experiments on three real-world datasets demonstrate that the MCKG has a significant improvement over state-of-the-art recommendation methods. Further ablation experiments verify the importance of multi-space fusion and geometry-aware optimization strategy, justifying the rationality and effectiveness of MCKG.

∗ Corresponding author.

† Fuzhen Zhuang is also at Zhongguancun Laboratory, Beijing, China

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

RecSys '23, September 18-22, 2023, Singapore, Singapore

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0241-9/23/09...$15.00

Relations r1 : interact  r 2 : directed  r 3 : acted  r4: genre

<!-- image -->

Figure 1: Tree and cyclic structures correspond to their most suitable modeling spaces: the hyperbolic and spherical spaces.

data, while the spherical space is superior for representing cyclic patterns.

Since complex KG usually have varied structures coupled together, choosing the ideal space is especially tricky. Constant curvature space methods [1, 5, 8, 28] just project entities to a single space, as a result, they fail to completely express the sophisticated structures of KG. Recently, some mixed-curvature space works [9, 13, 33] merge distinct spaces, aiming to characterize various structures. Unfortunately, how to find the most effective combination of these manifolds is also a challenging task. The simplest method [40] is to list all possible combinations and choose the one that best utilizes these geometry. Obviously, this method would be efficient but impractical due to the time-consuming and low-scalability [25]. Another solution [5] is to treat the curvature 𝜅 as a trainable parameter, aiming to learn the optimal curvature from the training data. However, there are still the following issues: 1) the constraints of model parameters depend on the changes of curvature. Once the curvature 𝜅 varies (e.g., from negative to positive), parameters may no longer satisfy the constraints. As a result, the internal structure of manifold will be destroyed, thus unable to obtain high-quality embeddings. 2) Since the manifolds belong to different spaces, i.e., M ∈ H , E , S , directly operating on various manifolds without considering the heterogeneity is problematic. For instance, the global distance computation just sums up the distances without distinguishing the importance of various manifolds.

Motivated by the above observations, in this paper, we propose a knowledge-based multiple adaptive spaces fusion method for recommendation (MCKG). Unlike existing methods [5, 26, 28, 37] that solely consider a specific manifold, we introduce a unified space U that is compatible with the hyperbolic, euclidean and spherical spaces. Note that the unified space can interpolate smoothly between all geometries of constant curvature, thus the internal structure of the manifold is preserved when training curvature. To break the limitation of the single space's expression ability, we further integrate multiple unified spaces to more accurately capture the structural information on a global level. On the other hand, we propose a geometry-aware optimization strategy that enables the pull and push processes benefited from both hyperbolic and spherical spaces. Specifically, in hyperbolic space, we set smaller margins in the area near to the origin, which is conducive to distinguishing between highly similar positive items and negative ones. At the same time, we set larger margins in the area far from the origin to ensure that the model has sufficient error tolerance. Conversely, in spherical space, the geometry-aware strategy assigns larger margins near to the origin and smaller margins far from the origin. Empirically, we conduct comprehensive experiments on the three benchmark datasets, the results show that MCKG outperforms the state-of-the-art recommendation methods. In summary, this work makes the following contributions:

- To the best of our knowledge, this is the first work to apply multiple adaptive spaces fusion for knowledge-enhanced recommender systems.
- We present a knowledge-based multiple adaptive spaces fusion method for recommendation. To obtain the high-quality embeddings for recommendation, we introduce the unified space to describe the complex structures of KG, then further fuse multiple subspaces in an attention manner. Finally, our proposed geometry-aware optimization strategy is the first work that considers the properties of both hyperbolic and spherical spaces.
- We conduct extensive experiments on the three benchmark datasets, results show that MCKG outperforms the state-ofthe-art recommendation methods.

## 2 PRELIMINARIES AND PROBLEM FORMULATION

## 2.1 Mathematical Concepts

In this subsection, we summarize preliminary notations for readers to better understand this paper.

Manifold : A manifold M of dimension 𝑛 is a topological where each point's neighborhood can be approximated by euclidean space R 𝑛 . For example, the earth can be modeled by the spherical space, and its local place can be estimated by R 2 . The notion of manifold is a generalization of the notion of surface.

Tangent space : For each point 𝑥 ∈ M , the tangent space T 𝑥 M of M at 𝑥 is defined as a 𝑛 -dimensional space estimating M around x at a first order.

Geodesics distance : Geodesics distance is the the generalization of a straight line in the Euclidean space, which indicates the shortest path between pairs of points.

Exponential map : The exponential map carries a vector 𝑣 ∈ T 𝑥 M of a point 𝑥 ∈ M to the manifold M , i.e., Exp 𝑥 : T 𝑥 M→M by traveling a fixed distance along the geodesic defined as 𝛾 ( 0 ) = 𝑥 with direction 𝛾 ′ ( 0 ) = 𝑣 . Each manifold has its own manner to design the exponential maps.

Logarithmic map : The logarithmic map is the reverse operation of the exponential map, which projects a point 𝑧 ∈ M on the manifold back to the tangent space of another point 𝑥 ∈ M , i.e., Log 𝑥 : M→T 𝑥 M . Similar to Exp 𝑥 , different manifolds correspond to their distinct logarithmic map.

## 2.2 Constant Curvature Spaces

Withthe constant sectional curvature 𝜅 , the three types of manifolds are defined as follows:

$$\mathcal { M } _ { \kappa } ^ { n } & = \begin{cases} \ k n _ { \kappa } ^ { n } \{ x \in \mathbb { R } ^ { n + 1 } \colon ( x , x ) _ { \kappa } = 1 / \kappa , x _ { 0 } > 0 \} & \text {for } \kappa < 0 \\ \ E _ { \kappa } \colon \mathbb { R } ^ { d } & \text {for } \kappa = 0 \\ \ S _ { \kappa } ^ { n } \{ x \in \mathbb { R } ^ { n + 1 } \colon ( x , x ) _ { \kappa } = 1 / \kappa \} & \text {for } \kappa > 0 \\ \end{cases} \\ \text {where the } / _ { \kappa } \, \cdot \, \rangle _ { \ } \text { represents the inner product} \colon$$

where the ⟨· , ·⟩ 𝜅 represents the inner product:

$$\langle x , y \rangle _ { \kappa } = \begin{cases} \sum _ { i = 0 } ^ { n } x _ { i } y _ { i } & \text {for } \kappa > 0 \\ - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { n } x _ { i } y _ { i } & \text {for } \kappa < 0 \end{cases} ( 2 )$$

Because hyperbolic or spherical spaces are not vector spaces, the vector operations (e.g., addition, subtraction and scalar multiplication) cannot be achieved. Therefore, we need to convert nonEuclidean spaces into tangent spaces for corresponding vector calculations. Specifically, the tangent space T 𝑥 M at point 𝑥 on M is a 𝑑 -dimensional Euclidean space that approximates M around 𝑥 :

$$\mathcal { T } _ { x } \mathcal { M } _ { \kappa } ^ { m } = \{ v \in \mathbb { R } ^ { n + 1 } \colon \langle v , x \rangle _ { \kappa } = 0 \} . \quad ( 3 ) \quad \text {In}$$

As mentioned in the previous section, the mapping between manifold M 𝑛 𝜅 ∈ { H 𝑛 𝜅 , S 𝑛 𝜅 } and its tangent space T 𝑥 M 𝜅 can be achieved by the exponential and logarithmic operations.

To connect vectors in tangent spaces, the parallel transport [14] defines a manner of transporting the local geometry along smooth curves that preserves the metric tensors. Specifically, for vector 𝑢 ∈ M and 𝑣 ∈ M , the parallel transport PT 𝑢 → 𝑣 : T 𝑢 M→T 𝑣 M carries a vector in T 𝑢 M along the geodesic from 𝑢 to 𝑣 .

## 2.3 Mix-Curvature Spaces

Since the constant curvature space is only suitable for a certain geometric structure, mix-curvature methods [1, 9, 17, 25, 40] expect to match the complicated geometry of data thus provide higher quality representations. To capture a wider range of curvatures, mixcurvature methods propose embeddings into product spaces [9] where each component has constant curvature. Formally, assuming a series of 𝑁 distinct constant curvature spaces M ( 1 ) , M ( 2 ) , ..., M ( 𝑁 ) the mix-curvature manifold are defined as:

$$\mathcal { M } = \mathcal { M } ^ { ( 1 ) } \times \mathcal { M } ^ { ( 2 ) } \times \dots \times \mathcal { M } ^ { ( N ) } , \quad \ \ ( 4 ) \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

where the × denotes Cartesian product.

## 2.4 Problem Formulation

Wefollow the recommendation setting [5, 30, 34]. Let 𝑈 = { 𝑢 1 , 𝑢 2 , ... } and 𝑉 = { 𝑣 1 , 𝑣 2 , ... } denote the sets of users and items, respectively. The user-item interaction matrix 𝑌 = { 𝑦 𝑢𝑣 | 𝑢 ∈ 𝑈,𝑣 ∈ 𝑉 } is defined based on the users' implicit feedback, where the value of 𝑦 𝑢𝑣 = 1 denotes that there is an interaction between user 𝑢 and item 𝑖 ; otherwise 𝑦 𝑢𝑣 = 0. We also have a knowledge graph 𝐺 available, which consists of massive entity-relation-entity triplets ( ℎ, 𝑟, 𝑡 ) , where ℎ ∈ 𝜙 , 𝑟 ∈ 𝜑 , and 𝑡 ∈ 𝜙 describe the head, relation, and tail of knowledge triples, and 𝜙 and 𝜑 denote the set of entities and relations. Given the user-item interactions and the item-side KG, the recommendation task is to train a RS model calculating the probability that the user 𝑢 will click item 𝑣 .

,

Table 1: Summary of operations in unified space U 𝑛 𝜅

| Name               | Operation                                                                                                    |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| Addition           | 𝑥 ⊕ 𝜅 𝑦 = ( 1 - 2 𝜅 ⟨ 𝑥,𝑦 ⟩- 𝜅 ∥ 𝑦 ∥ 2 2 ) 𝑥 +( 1 + 𝜅 ∥ 𝑦 ∥ 2 2 ) 𝑦 1 - 2 𝜅 ⟨ 𝑥,𝑦 ⟩+ 𝜅 2 ∥ 𝑥 ∥ 2 2 ∥ 𝑦 ∥ 2 2 |
| Multiplication     | 𝑥 ⊗ 𝜅 𝑦 = exp 𝜅 o ( 𝑥 · log 𝜅 o ( 𝑦 ))                                                                       |
| Concatenate        | 𝑥 ⊖ 𝜅 𝑦 = exp 𝜅 o ( 𝑥 &#124; &#124; log 𝜅 o ( 𝑦 ))                                                           |
| Dot Product        | 𝑥 ⊙ 𝜅 𝑦 = exp 𝜅 o ( 𝑥 𝑇 log 𝜅 o ( 𝑦 ))                                                                       |
| Geodesics Distance | 𝑑 𝜅 ( 𝑥,𝑦 ) = 2 tan - 1 𝜅 (∥ - 𝑥 ⊕ 𝜅 𝑦 ∥ 2 )                                                                 |
| Exponential Map    | exp 𝜅 𝑥 ( 𝑣 ) = 𝑥 ⊕ 𝜅 ( tan 𝜅 ( 𝜆 𝜅 𝑥 ∥ 𝑣 ∥ 2 2 ) 𝑣 ∥ 𝑣 ∥ 2                                                  |
| Logarithmic Map    | log 𝜅 𝑥 ( 𝑦 ) = 2 𝜆 𝜅 𝑥 tan - 1 𝜅 (∥ - 𝑥 ⊕ 𝜅 𝑦 ∥ 2 ) - 𝑥 ⊕ 𝜅 𝑦 ∥- 𝑥 ⊕ 𝜅 𝑦 ∥ 2                                |

## 3 METHOD

In this section we present the MCKG model in Figure 2. First, we introduce the 𝜅 -Stereographic manifold and the extraction of highorder information on a single manifold. Then, we fuse the each adaptive subspaces to fully capture the global structural information. Finally, we propose a novel geometry-aware optimization strategy to be compatible with both hyperbolic and spherical spaces.

## 3.1 Unifying All Curvatures

We adopt 𝜅 -Stereographic model [1] as the base manifold, which is a unification of constant curvature manifolds: H 𝑛 𝜅 , E 𝑛 𝜅 and S 𝑛 𝜅 . The model is a Riemannian manifold with constant sectional curvature 𝜅 and dimension 𝑛 :

$$\mathbb { U } _ { \kappa } ^ { n } = \left \{ \begin{array} { c c } x \in \mathbb { R } ^ { n } \colon \| x \| _ { 2 } < 1 / \sqrt { - \kappa } , \lambda _ { x } ^ { \kappa } & \kappa < 0 \\ \mathbb { R } ^ { n } , \lambda _ { x } ^ { \kappa } & \kappa \geq 0 \end{array} \right \}$$

where 𝜆 𝜅 𝑥 = 2 1 + 𝜅 ∥ 𝑥 ∥ 2 2 is conformal metric tensor at point 𝑥 . For the 𝜅 -Stereographic model, the optimization step changes the geometry of space and the constraints of parameters, which also depend on the curvature. Fortunately, when we use curvature 𝜅 as a trainable parameter, the model can interpolate smoothly between all geometries of constant curvature, and the detailed proof process can refer to paper [1].

The basic operations of the unified space U 𝑛 𝜅 are summarized in Table 1, but the trigonometric functions in Table 1 are not complete, here we complement its definition.

$$t a n _ { K } ( x ) & = \begin{cases} \kappa ^ { - 1 / 2 } t a n ( x \kappa ^ { 1 / 2 } ) & \kappa > 0 \\ x + \frac { 1 } { 3 } \kappa x ^ { 3 } & \kappa = 0 \\ | \kappa | ^ { - 1 / 2 } t a n h ( x | \kappa | ^ { 1 / 2 } ) & \kappa < 0 \end{cases}$$

$$t a n _ { \kappa } ^ { - 1 } ( x ) = \begin{cases} \kappa ^ { - 1 / 2 } t a n ^ { - 1 } ( x \kappa ^ { 1 / 2 } ) & \kappa > 0 \\ \, x - \frac { 1 } { 3 } \kappa x ^ { 3 } & \kappa = 0 \\ | \kappa | ^ { - 1 / 2 } t a n h ^ { - 1 } ( x | \kappa | ^ { 1 / 2 } ) & \kappa < 0 \end{cases}$$



Figure 2: Overview of MCKG model architecture: (A) The overall process of high-order information extraction based on the single adaptive space U 𝑛 𝜅 ; (B) Aggregating the subspace embeddings and adopt an attention mechanism to calculate the global distance. (C) The proposed optimization strategy enables the pull and push processes to be geometric-aware in both hyperbolic and spherical spaces.

<!-- image -->

## 3.2 High-Order Information Extraction

neighbors aggregati Since the initial embeddings are in Euclidean space, we explicitly encode the Euclidean embeddings onto the unified space before feeding the subsequent layers. Denote the initial user, item and relation embeddings as 𝑥 𝑢 , 𝑥 𝑣 and 𝑥 𝑟 respectively, then the exponential maps are applied:

$$e _ { u } = e x p _ { 0 } ^ { \kappa } ( x _ { u } ) , \ \ e _ { v } = e x p _ { 0 } ^ { \kappa } ( x _ { v } ) , \ \ e _ { r } = e x p _ { 0 } ^ { \kappa } ( x _ { r } ) . \quad ( 8 )$$

1 NN2 ev We present a relational attention function as follows to describe user's interest:

Avera

$$c _ { u } ^ { r } = e _ { u } \odot _ { \kappa } e _ { r } .$$

(1)

$$& \text {as} \\ & \quad e _ { s ( v ) } ^ { u } = \sum _ { \ a \in s ( v ) } \hat { c } _ { u } ^ { r } \otimes _ { x } e _ { a } , \quad \hat { c } _ { u } ^ { r } = \frac { \exp ( c _ { u } ^ { r _ { u , a } } ) } { \sum _ { \ a \in s ( v ) } \exp ( c _ { u } ^ { r _ { v , a } } ) } , \quad ( 1 0 ) \\ & \text {where } s ( v ) \text { denotes the selected neighbors of } v , \, \text {the } \hat { c } _ { u } ^ { r } \text { is the normal-}$$

neighbors Unified (b) ev Intuitively, 𝑐 𝑟 𝑢 means the importance of relation 𝑟 to user 𝑢 . The neighboring information can be continuously aggregated based on 𝑐 𝑟 𝑢 . Specifically, the neighborhood embedding of entity 𝑣 is defined as

v3 v4 where 𝑠 ( 𝑣 ) denotes the selected neighbors of 𝑣 , the ˆ 𝑐 𝑟 𝑢 is the normalized user-relation score, and 𝐸𝑥𝑝 (·) means the basic exponential function.

The key idea of Graph Convolution Network (GCN) is to aggregate feature information from a node's neighbors. Obviously, feeding the entire knowledge graph to model will greatly increase the computational burden. To this end, we assign a fixed size receptive field for each node to control the amount of calculation.

After getting the neighborhood representation, we aim to represent target node at the next layer. There are three kinds of aggregators:

- GCN Aggregator [20] adds the two representations and performs a nonlinear transformation, and it can be formulated as

$$e _ { v } ^ { ( k + 1 ) } = e x p _ { v } ^ { \kappa } ( \sigma l o g _ { v } ^ { \kappa } ( W \otimes _ { \kappa } ( e _ { v } ^ { ( k ) } \oplus _ { \kappa } e _ { s ( v ) } ^ { ( k ) } ) \oplus _ { \kappa } b ) ) , \quad ( 1 1 )$$

Layer

Layer 1 … k multiply and accumulate where 𝜎 is LeakyRelu activation function, 𝑊 and b represent the weight matrix and bias respectively, ⊗ 𝜅 and ⊕ 𝜅 denote the multiplication and addition as mentioned in Table 1.

- k-order neighbors of v ... pairwise · GraphSage Aggregator [12] concatenates the two representations, then fed to the activation function:

$$e _ { v } ^ { ( k + 1 ) } = e x p _ { v } ^ { \kappa } ( \sigma l o g _ { v } ^ { \kappa } ( W \otimes _ { \kappa } ( e _ { v } ^ { ( k ) } \ominus _ { \kappa } e _ { s ( v ) } ^ { ( k ) } ) \oplus _ { \kappa } b ) ) , \quad ( 1 2 )$$

where the ⊖ is the concatenation operation.

- ev NN2 · Neighbor Aggregator [34] takes the neighborhood representation of entity 𝑣 as the output representation, and it can be formulated as

(c)

$$e _ { v } ^ { ( k + 1 ) } = e x p _ { v } ^ { \kappa } ( \sigma l o g _ { v } ^ { \kappa } ( W \otimes _ { \kappa } ( e _ { s ( v ) } ^ { ( k ) } ) \oplus _ { \kappa } b ) ) . \\$$

The embeddings at different layers capture different semantics. E.g., the first layer focuses on items that users has directly interacted with, the second layer emphasizes overlapping items between users, and the higher-layers can capture higher-order information [35]. After 𝐾 layers aggregation, we further combine the embeddings obtained at each layer to form the final item representations:

$$e _ { v } ^ { * } = e _ { v } ^ { ( 0 ) } \oplus _ { \kappa } e _ { v } ^ { ( 1 ) } \dots \oplus _ { \kappa } e _ { v } ^ { ( k ) } , e _ { u } ^ { * } = e _ { u } ^ { ( 0 ) } .$$

## 3.3 Multiple Spaces Fusion

Since the complicated structures are not evenly distributed over the entire graph, even if a single space can learn the ideal curvature 𝜅 , it is still not capable of fully modeling the entire graph. Now we consider a sequence of manifolds U ( 1 ) , U ( 2 ) , ..., U ( 𝑀 ) , where the item (user) embedding in the 𝑚 -th subspace is denoted as 𝑒 ∗ ,𝑚 𝑣 . To accurately capture the global structural information, we design a fusion mechanism by aggregating subspace embeddings. First, we take the average of subspace representations as the global fused embeddings 𝑒 ∗ ,𝑔 𝑣 , then we concatenate the average value and the original subspace embedding to obtain the new subspace embedding:

$$e _ { v } ^ { * , g } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } e _ { v } ^ { * , m } , \\ e _ { v } ^ { * , m } = e _ { v } ^ { * , m } \ominus _ { K } e _ { v } ^ { * , g } .$$

As a result, the new embeddings take into account the global information, which is not limited to the current space. Finally, we further integrate the series of subspaces in an attention-enhanced manner. In other words, we calculate the global distance in an attention manner. Formally, the global distance between user 𝑢 and item 𝑣 is the weighted sum of the distances in each subspace:

$$d i s t ( u , v ) = \sum _ { m = 1 } ^ { M } w ( e _ { u } ^ { * , m } , e _ { v } ^ { * , m } ) d _ { K } ( e _ { u } ^ { * , m } , e _ { v } ^ { * , m } ) , \quad ( 1 6 ) \quad \text {awward} \quad \text {is} \quad \sum _ { \substack { \cdot \\ \cdot \cdot } } ^ { M } ( 1 6 )$$

where the 𝑑 𝜅 (·) represents geodesics distance as mentioned in Table 1, and the 𝑤 ( 𝑒 ∗ ,𝑚 𝑢 , 𝑒 ∗ ,𝑚 𝑣 ) is defined as:

$$w ( e _ { u } ^ { * , m } , e _ { v } ^ { * , m } ) = w ^ { \prime } ( e _ { u } ^ { * , m } ) + w ^ { \prime } ( e _ { v } ^ { * , m } ) .$$

Here the importance of a user-item pair is the sum of the influence of 𝑢 and 𝑣 . Due to the 𝑤 ′ ( 𝑒 ∗ ,𝑚 𝑢 ) and 𝑤 ′ ( 𝑒 ∗ ,𝑚 𝑣 ) are the entity level subspace attention, the weights can be pre-trained as follows:

$$w ^ { \prime } ( e _ { u } ^ { * , m } ) = \frac { e x p ( \alpha _ { u } ^ { m } ) } { \sum _ { i = 1 } ^ { M } e x p ( \alpha _ { u } ^ { i } ) } , w ^ { \prime } ( e _ { v } ^ { * , m } ) = \frac { e x p ( \alpha _ { v } ^ { m } ) } { \sum _ { i = 1 } ^ { M } e x p ( \alpha _ { v } ^ { i } ) } , \quad ( 1 8 ) \quad \text {type}$$

where the 𝛼 𝑢 and 𝛼 𝑣 are matrices of size 1 × M and each entry stands for the importance of the corresponding subspace.

$$\alpha _ { u } = W [ e _ { u } ^ { * , 1 } \ominus _ { \kappa } \dots \ominus _ { \kappa } e _ { u } ^ { * , M } ] , \alpha _ { v } = W [ e _ { v } ^ { * , 1 } \ominus _ { \kappa } \dots \ominus _ { \kappa } e _ { v } ^ { * , M } ] .$$

In summarize, the multiple spaces fusion method has the following advantages:

- Different from existing mixed curvature methods [9, 17, 33, 40] manually combine various manifolds, we adopt the adaptive curvature spaces U 𝑛 𝜅 instead of constant curvature spaces, e.g., H 𝑛 𝜅 , E 𝑛 𝜅 and S 𝑛 𝜅 .
- To accurately model the complex structure of the graph, we further integrate each subspace in an attention-enhanced manner from a global perspective.

## 3.4 Geometry-aware Optimization Strategy

The margin-based ranking loss has shown to be quite beneficial for non-Euclidean recommendation models [26, 37]. This loss seeks to distinguish user-item pairs up to a specified margin into positive and negative samples, once the margin is satisfied the pairs are regarded as well separated. Specifically, for each user 𝑢 we sample a positive item 𝑖 and a negative item 𝑗 , and the margin loss is defined as

$$\mathcal { L } ( u , i , j ) = \max ( \underbrace { d i s t ^ { 2 } ( u , i ) } _ { \ p u l l } - \underbrace { d i s t ^ { 2 } ( u , j ) } _ { \ p u s h } + m , 0 ) ,$$

where the 𝑑𝑖𝑠𝑡 (·) denotes the global distance in unified space as mentioned in equation (16), 𝑚 is the margin between ( 𝑢, 𝑖 ) and ( 𝑢, 𝑗 ) . As a result, positive items are pulled closer to user while negative items are pushed outside the margin.

However, it is still unclear how to pick a suitable margin. HGCF [26] sets the margin to a constant, this action ignores the exponentially expansive geometric properties of hyperbolic spaces. Furthermore, to emphasize the importance of modeling tail items, HICF [37] sets a greater margin in the vicinity of the hyperbolic origin and a smaller margin elsewhere. Although this method can alleviate the power-law distribution to a certain extent, it will hurt the model performance. E.g., in hyperbolic spaces, the space capacity of the area near the origin is extremely narrow, thus a smaller margin is beneficial to distinguish highly similar entities. While in areas far away from the origin, we prefer a greater margin because the room is spacious enough. On the other hand, in spherical spaces, the origin area is relatively spacious, but the boundary area is narrow. Therefore, we expect assigning larger margins to the area closer to the spherical origin, and smaller margins to the area farther from the spherical origin.

Motivated by the above considerations, we aim to devise a geometricaware optimization strategy which can benefit from both hyperbolic and spherical spaces. Paper [24] discovered an intriguing phenomenon: when entities 𝑥 and 𝑦 progressively moves away from the origin in hyperbolic space, the ratio 𝑑𝑖𝑠𝑡 ( 𝑥,𝑦 ) 𝑑𝑖𝑠𝑡 ( 𝑥, o )+ 𝑑𝑖𝑠𝑡 ( 𝑦, o ) is getting greater. Obviously, the reason is that the numerator (distance) grows significantly faster than the denominator (radius) in hyperbolic spaces. On the contrary, as entities 𝑥 and 𝑦 move away from the spherical origin, the ratio 𝑑𝑖𝑠𝑡 ( 𝑥,𝑦 ) 𝑑𝑖𝑠𝑡 ( 𝑥, o )+ 𝑑𝑖𝑠𝑡 ( 𝑦, o ) will gradually become smaller. Surprisingly, the change trend of this ratio is in line with our expectations for the margin. Inspired by these observations, we carefully design the margin 𝑚 𝑔 as follows:

$$m _ { g } = \sigma ( \frac { d i s t ( u , i ) } { d i s t ( u , 0 ) + d i s t ( i , 0 ) } ) + c ,$$

where 𝜎 is the sigmoid function for normalization, and we maintain the constant 𝑐 for better parameter tuning. Then the 𝑚 𝑔 is utilized to replace the 𝑚 in equation (20). As a result, the margin will increase as it slowly departs from the origin in hyperbolic spaces, and in spherical spaces it's the exact opposite. To summarize, either in the hyperbolic or spherical spaces, our model have the ability to perceive the spatial structures and learn the ideal margin.

It is worth mentioning that HICF [37] aims to improve the attention of tail items in hyperbolic space, thus the margin 𝑚 ℎ is set as follows:

$$m _ { h } = \sigma ( d i s t ( u , o ) + d i s t ( i , o ) - ( d i s t ( u , i ) ) + c .$$

Therefore, it assigns larger margins in narrower areas and smaller margins in spacious areas, exactly the opposite of what we expect. We will compare the performance of various margins in Section 4.3.1.

## 3.5 Time Complexity Analysis

As we can see, the layer-wise propagation is the core operation in MCKG. Let 𝑁 denotes the number of user-item interactions, for the 𝑘 -th layer, the matrix multiplication computational complexity is 𝑂 ( 𝑁 · | 𝑠 ( 𝑣 )| · 𝑑 𝑘 ) , where 𝑠 ( 𝑣 ) represents the item neighbor sampling size, and 𝑑 𝑘 denotes the current transformation size. As a result, the overall time complexity of our proposed model is 𝑂 ( ˝ 𝑀 𝑚 = 1 ˝ 𝐾 𝑘 = 1 𝑁 · | 𝑠 ( 𝑣 )| · 𝑑 𝑘 ) , where 𝑀 is the number of manifolds, and 𝐾 denotes the number of graph convolution layers. Parameter setting details will be introduced in Section 4.1.4.

## 4 EXPERIMENTS

In this section, we introduce the experimental settings, and perform extensive comparative experiments with the state-of-the-art models. Then we conduct comprehensive ablation studies to verify the designs in MCKG and reveal the reasons of its effectiveness. Finally, we visualize the learned embedding and analyze the performance of the recommendation model in data-sparse scenarios.

## 4.1 Experimental Settings

4.1.1 Datasets. We perform extensive experiments on the three real-world datasets to assess the performance of the MCKG algorithm. Table 1 summarizes the statistics of datasets.

Movielens-1M 1 : This movie dataset has been extensively applied to testify recommendation methods. We adopt the version with one million interactions, and each user has more than 20 ratings.

LastFM 2 : The LastFM online music system, which offers the listening count as weight for each user-item interaction, provided the dataset from a set of about 2,000 users.

Book-Crossing 3 : This dataset provided by the Book-Crossing website, which contains the evaluation of books by more than 10,000 users.

4.1.2 Baselines. To demonstrate the effectiveness, we compare our proposed MCKG algorithm with the following SOTA methods.

- HGCF [26]: The first work that proposes a hyperbolic GCN model for collaborative filtering. Specifically, HGCF utilizes the Lorentz representation to initialize user and item embeddings, performs graph convolution in the tangent space, and projects back to hyperbolic space to learn the final embeddings.
- RippleNet [30]: This is the first work to propose the conception of preference propagation. In particular, RippleNet takes the item clicked by the user as the origin, expands

1 https://grouplens.org/datasets/movielens/

2 https://grouplens.org/datasets/hetrec-2011/

3 http://www2.informatik.uni-freiburg.de/ cziegler/BX/

around like the water ripples, and constantly absorbs 1-hop and 2-hop neighbors to spread information.

- KGAT [34]: KGAT trains the embeddings of target node based on its neighbors embeddings, and further recursively executes the embeddings propagation to obtain the highorder connectivity in KG. Additionally, it design an attention mechanism is employed to learn the weight of each neighbor during the information propagation.
- KBHP [28]: KBHP is the first work to combine KG and hyperbolic space for recommendation, which is the hyperbolic version of RippleNet.
- LKGR [5]: To better model the scale-free tripartite graphs and investigate the intrinsic hierarchical graph structures, LKGR employs different information propagation strategies in the hyperbolic space to explicitly encode the historical interactions and KG.

4.1.3 Evaluation Metrics. For each user, we randomly pick one item that user has interacted with, and sample 100 unobserved or negative items to construct the test sets. We adopt two common ranking evaluation metrics, i.e., Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) to evaluate all recommendation methods. As a result, the 𝐻𝑅 @ 𝐾 clearly assesses whether the test item is appeared on the top-K list, while the 𝑁𝐷𝐶𝐺 @ 𝐾 assigns higher points to items at the top of the hit list, to emphasize the importance of their positions.

4.1.4 Parameter Settings. In the experiments, we randomly pick 70% of each dataset as the train set, the rest 30% as the test set. For the baseline methods, the model parameters are in accordance with the authors' paper. We set the sampling size and the hop as: 8 and 3 for Movielens-1M and Book-Crossing; 4 and 3 for LastFM. The number of manifolds defaults to 3. Moreover, if HR and NDCG do not increase for 20 successive epochs on the test set, the early stopping strategy is implemented.

## 4.2 Overall Performance Comparison

Table 3 shows the performance (HR@10, HR@20, NDCG@10 and NDCG@20) of all competitive algorithms. The observations obtained from Table 3 are as follows:

- Most KG-enhanced methods perform better than HGCF, which proves that the iterative aggregation is an effective way to extract high-order information from KGs. Note that HGCF outperforms RippleNet in the Movielens-1M, one possible reason is that the Movie datasets have a relatively sufficient user-item interactions, so the improvement from KGs is not particularly obvious.
- Under the same conditions combined with KGs, the hyperbolic model (e.g., KBHP and LKGR) performs better than the Euclidean model (e.g., RippleNet and KGAT). It is crucial to note that KBHP is the hyperbolic version of RippleNet, we can see that in each datasets, KBHP performs markedly better than RippleNet.
- Compared with other competitive modles, MCKG performs best in most cases. Roughly speaking, this is attributed to our model's ability of adaptive curvature learning and geometry

Table 2: Three datasets used in this paper.

| Datasets              | Movielens-1M   | LastFM      | Book-Crossing   |
|-----------------------|----------------|-------------|-----------------|
| Users/Items           | 6,036/2,347    | 1,872/3,846 | 17,860/14,910   |
| Interactions          | 753,772        | 42,346      | 139,746         |
| KG Entities/Relations | 16,954/32      | 9,366/60    | 25,787/18       |
| KG triples            | 20,195         | 15,518      | 60,787          |

Figure 3: The performance compared with other competing methods on the three datasets, with latent dimension ranging from 2 to 32.

<!-- image -->

Table 3: The overall comparison among all competing models, the best results of all methods are in bold, while the second-best results are underlined.

| Datasets   | Movielens-1M   | Movielens-1M   | Movielens-1M   | Movielens-1M   | LastFM   | LastFM   | LastFM   | LastFM   | Book-Crossing   | Book-Crossing   | Book-Crossing   | Book-Crossing   |
|------------|----------------|----------------|----------------|----------------|----------|----------|----------|----------|-----------------|-----------------|-----------------|-----------------|
| Methods    | H@10           | H@20           | N@10           | N@20           | H@10     | H@20     | N@10     | N@20     | H@10            | H@20            | N@10            | N@20            |
| HGCF       | 0.602          | 0.763          | 0.376          | 0.392          | 0.544    | 0.596    | 0.350    | 0.361    | 0.336           | 0.528           | 0.251           | 0.340           |
| RippleNet  | 0.596          | 0.752          | 0.372          | 0.391          | 0.562    | 0.611    | 0.361    | 0.372    | 0.368           | 0.547           | 0.281           | 0.359           |
| KGAT       | 0.615          | 0.778          | 0.394          | 0.407          | 0.571    | 0.614    | 0.364    | 0.377    | 0.379           | 0.551           | 0.309           | 0.356           |
| KBHP       | 0.628          | 0.781          | 0.391          | 0.408          | 0.612    | 0.645    | 0.387    | 0.406    | 0.398           | 0.609           | 0.344           | 0.390           |
| LKGR       | 0.640          | 0.791          | 0.395          | 0.411          | 0.628    | 0.678    | 0.396    | 0.418    | 0.380           | 0.582           | 0.324           | 0.364           |
| MCKG       | 0.645          | 0.802          | 0.401          | 0.418          | 0.635    | 0.691    | 0.405    | 0.432    | 0.402           | 0.603           | 0.348           | 0.392           |
| Improv.    | 0.7%           | 1.3%           | 1.5%           | 1.7%           | 1.1%     | 1.9%     | 2.2%     | 3.3%     | 1.0%            | -1.0%           | 1.1%            | 0.5%            |

perception. We will reveal this phenomenon in the ablation study.

Figure 3 shows the performance of all the competitive algorithms with different numbers of dimension 𝑑 . With the latent dimension changes, the performance of all models fluctuates within a small range. From this figure, we conclude that Euclidean methods are easily impacted by the embedding size, while hyperbolic methods still perform well even in low dimensions. Specifically, MCKG performs the best in most situations. It is worth mentioning that LKGR also performs relatively well. In addition, KBHP has an excellent performance on Book-Crossing dataset.

Table 4: The comparison of performance (HR@20 and NDCG@20) among different variants.

| Datasets Aggregators   | Movielens-1M   | Movielens-1M   | LastFM   | LastFM   | Book-Crossing   | Book-Crossing   |
|------------------------|----------------|----------------|----------|----------|-----------------|-----------------|
| Datasets Aggregators   | HR             | NDCG           | HR       | NDCG     | HR              | NDCG            |
| GCN-c                  | 0.783          | 0.402          | 0.682    | 0.411    | 0.534           | 0.356           |
| GCN-h                  | 0.770          | 0.393          | 0.675    | 0.401    | 0.503           | 0.332           |
| GCN-g                  | 0.802          | 0.418          | 0.691    | 0.432    | 0.579           | 0.374           |
| GraphSage-c            | 0.754          | 0.384          | 0.638    | 0.406    | 0.576           | 0.364           |
| GraphSage-h            | 0.730          | 0.361          | 0.613    | 0.372    | 0.568           | 0.352           |
| GraphSage-g            | 0.771          | 0.392          | 0.642    | 0.415    | 0.609           | 0.392           |
| Neighbor-c             | 0.742          | 0.354          | 0.593    | 0.351    | 0.524           | 0.343           |
| Neighbor-h             | 0.733          | 0.338          | 0.561    | 0.325    | 0.501           | 0.327           |
| Neighbor-g             | 0.767          | 0.365          | 0.602    | 0.362    | 0.566           | 0.366           |

## 4.3 Ablation Study

We conduct a comprehensive ablation study on MCKG by showing how the model components affect its performance.

4.3.1 Impact of aggregators and geometry-aware margin. To further investigate the influence of aggregators (see the equation (12) (13) (14)) and margins on the model, we conduct a comprehensive experiment and the results are shown in Table 4, where the suffix '-c' means we adopt constant c as the margin, '-g' indicates the geometry-aware margin in MCKG (see the equation (21)), and '-h' represents the margin in HICF (see the equation (22)). From this table, we have the following observations:

- The model depends greatly on the aggregator selection. GCN and GraphSage aggregators perform substantially better than Neighbor, one possible reason is that the Neighbor aggregator ignores its own information.
- All aggregators adopting geometry-aware margins '-g' are significantly better than other margins (i.e., '-c' and '-h'), and the margin performance in HICF is not even as good as a constant margin. This shows that HICF can alleviate the power-law distribution to a certain extent, it will hurt the model performance.
- To summarize, the choice of aggregators is still important, and the margin we designed will significantly improve the model performance, especially in larger dataset (e.g., BookCrossing).

4.3.2 Impact of KG propagation depth. We consider the aggregation depth from 𝑑 = { 1 , 2 , 3 } , the results show that 𝑑 = 2 performs best on Movielens-1M, while 𝑑 = 1 works best on LastFM and Book-Crossing. We can also see that the model performs best when 𝑑 = 1 or 2, but when 𝑑 = 3, all metrics indicate to a rapid collapse.

Table 5: The comparison of performance (HR@20 and NDCG@20) among different depths.

| Datasets depths   | Movielens-1M   | Movielens-1M   | LastFM   | LastFM   | Book-Crossing   | Book-Crossing   |
|-------------------|----------------|----------------|----------|----------|-----------------|-----------------|
| Datasets depths   | HR             | NDCG           | HR       | NDCG     | HR              | NDCG            |
| 1                 | 0.783          | 0.402          | 0.691    | 0.432    | 0.534           | 0.356           |
| 2                 | 0.802          | 0.418          | 0.684    | 0.428    | 0.609           | 0.392           |
| 3                 | 0.787          | 0.407          | 0.662    | 0.414    | 0.601           | 0.376           |

Table 6: The comparison of performance (HR@20 and NDCG@20) among different numbers of manifolds.

| Datasets numbers   | Movielens-1M   | Movielens-1M   | LastFM   | LastFM   | Book-Crossing   | Book-Crossing   |
|--------------------|----------------|----------------|----------|----------|-----------------|-----------------|
| Datasets numbers   | HR             | NDCG           | HR       | NDCG     | HR              | NDCG            |
| 1                  | 0.778          | 0.409          | 0.663    | 0.401    | 0.573           | 0.356           |
| 2                  | 0.793          | 0.414          | 0.685    | 0.430    | 0.609           | 0.390           |
| 3                  | 0.802          | 0.418          | 0.691    | 0.432    | 0.608           | 0.392           |
| 4                  | 0.800          | 0.417          | 0.688    | 0.430    | 0.609           | 0.392           |

Essentially, the appropriate depth of aggregation can improve performance, but too long relationship chains are more likely to bring noise.

4.3.3 Impact of Muti-space Fusion. We consider the number of manifold from { 1 , 2 , 3 , 4 } and the results are summarized in Table 6, note that there is no space fusion when 𝑛𝑢𝑚𝑏𝑒𝑟 = 1. Obviously the multi-space fusion is significantly better than the single space. When the 𝑛𝑢𝑚𝑏𝑒𝑟 = { 2 , 3 , 4 } , the performance is stronger than the case of 𝑛𝑢𝑚𝑏𝑒𝑟 = 1. In addition, the performance of the model is similar in the case of number is 2, 3, or 4. One possible reason is that although the dataset is sparse, the distribution is not particularly complex. As a result, when the number of spaces increases, the performance does not change much.

## 4.4 Embedding Visualization

We used t-SNE to visualize the item embeddings as shown in Figure 4. Similar to HGCF [26], to better understand the influence of graph convolutions we depict item embeddings before and after the MCKG layers. Furthermore, the items are divided into three groups based on the prevalence of interactions, and we color each tertile independently.

Before performing MCKG, we can see that the three groups of item embeddings are evenly distributed in a sphere area. After feeding the three datasets to MCKG, we have the following observations:

- In Movielens-1M, we can clearly see that unpopular items (purple points) are pushed out at both ends, while the popular items (green and yellow points) are pulled in the middle region. This may be caused by the geometry-aware optimization strategy. Specifically, when the curvature is trainable, our model can distinguish items by popularity, whether in hyperbolic or spherical spaces.

Figure 4: Item embeddings visualisation in the unified space U before and after the MCKG graph convolutioanl layers.

<!-- image -->

- In LastFM, we hardly see a clear difference between the trained item embeddings and the original embeddings, but this may be more beneficial for the recommendation's fairness. On the other hand, in Book-Crossing, we roughly see that the more popular items (green points) are distributed in the upper part of the sphere, and the unpopular items (purple points) are distributed in the lower part of the sphere.

## 4.5 Sparse Scene Study

Data sparsity is a serious problem in recommendation field, to study the performance of recommendation models in sparse circumstances, we change the training set ratio of Movielens-1M from 10% to 90%, the rest datasets as test set. From Table 7, compared with the ratio=90%, the HR@20 of each model decreases by 6.2%, 3.7%, 4.4%, 2.7%, 3.0%, 3.2% when ratio=10%.

With the change of ratio, HGCF fluctuates the most, which shows that compared to other KG-enhanced models, it is more susceptible to sparse data. This also confirms that using KG as auxiliary information is a effective strategy to improve recommendation performance in sparse scenarios. On the other hand, we found that the non-Euclidean methods (i.e., KBHP, LKGR and MCKG) perform more stable than the Euclidean methods (i.e., RippleNet and KGAT) in sparse scene. One possible reason is that due to the spatial characteristics of hyperbolic spaces, even in sparse scene, its ability to distinguish positive and negative items still be maintained.

## 5 RELATED WORK

In this section, we review two relevant prior works: KG-enhanced recommendation methods and the hyperbolic representation learning.

## 5.1 KG-enhanced Recommendation Methods

Currently, KG-enhanced recommendation methods has gradually attracted the attention of researchers, here we divide them into embedding-based methods, path-based methods and GCN-based methods:

Embedding-based approaches typically use knowledge graph embedding (KGE) methods to endow knowledge graph for entity vectors, and further feed these vectors to recommendation module [2]. For instance, to fully exploit the knowledge base, CKE [38] considers three components to extract semantic features from the item's structured content, textual content and visual content, respectively. Finally, CKE is formed to jointly learn the implicit vector of collaborative filtering and the semantics of items based on knowledge base. To dynamically adapt to user preference in real time, DKN [31] adopts a sequential learning method, that is, the knowledge embedding learning module and the recommendation system module are independent of each other. It uses semantic information and knowledge information to represent news, and makes predictions based on user historical behavior and current candidate news.

Path-based approaches take into account various relationships between items and users in the knowledge network, which aims to offer accurate directions for recommendation tasks. Since the KG

Table 7: Results of HR@20 on MovieLens-1M with different ratios of training set

| Model     | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | Ratios of training set   | %change.   |
|-----------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|------------|
| Model     | 10%                      | 20%                      | 30%                      | 40%                      | 50%                      | 60%                      | 70%                      | 80%                      | 90%                      | %change.   |
| HGCF      | 0.7183                   | 0.7320                   | 0.7346                   | 0.7488                   | 0.7503                   | 0.7538                   | 0.7631                   | 0.7597                   | 0.7633                   | 6.2%       |
| RippleNet | 0.7236                   | 0.7256                   | 0.7263                   | 0.7279                   | 0.7385                   | 0.7492                   | 0.7490                   | 0.7504                   | 0.7403                   | 3.7%       |
| KGAT      | 0.7456                   | 0.7648                   | 0.7637                   | 0.7646                   | 0.7648                   | 0.7729                   | 0.7781                   | 0.7786                   | 0.7791                   | 4.4%       |
| KBHP      | 0.7611                   | 0.7729                   | 0.7786                   | 0.7865                   | 0.7878                   | 0.7810                   | 0.7812                   | 0.7817                   | 0.7815                   | 2.7%       |
| LKGR      | 0.7690                   | 0.7697                   | 0.7804                   | 0.7806                   | 0.7808                   | 0.7910                   | 0.7914                   | 0.7905                   | 0.7918                   | 3.0%       |
| MCKG      | 0.7805                   | 0.7809                   | 0.7858                   | 0.7962                   | 0.7997                   | 0.8023                   | 0.8010                   | 0.8008                   | 0.8055                   | 3.2%       |

can construct heterogeneous information networks with user-item interactions, the conventional meta-path techniques can be applied to extracting heterogeneous information networks for recommendation. For instance, MCRec [15] learns representations of users, items and their interaction contexts, i.e., aggregated meta-paths. Specifically, MCRec uses a hierarchical neural network to model the meta-path context as a low-dimensional embedding, and enhances the three representations with a joint attention mechanism. Due to the introduction of meta-path-based context, this model not only has excellent performance, but also has a certain interpretability.

GCN-based methods typically employ iterative aggregation of the target node's neighbors to fully mine the high-level information of KG. For instance, RippleNet [30] takes the item clicked by the user as the origin, expands around like the water ripples, and constantly absorbs 1-hop and 2-hop neighbors to spread information. To automatically discover the semantic information of KG, KGCN [32] samples from the neighborhood of each entity in KG as its receptive field, and then combines the neighborhood information with bias when computing the representation for a given entity. The receptive field can be further extended to multi-hops to model high-order information and capture users' latent long-range interests. Similarly, KGAT [34] updates a node's embedding based on the embedding of its neighbors, and recursively performs this embedding propagation to capture high-order connectivity in linear time complexity. Additionally, an attention mechanism is employed to learn the weight of each neighbor during propagation.

## 5.2 Hyperbolic Representation Learning

Recently, Hyperbolic representation learning has taken an important role in RSs [5, 26, 28, 29, 39]. HyperML [29] studies metric learning in hyperbolic space and its connection to CF. Furthermore, HGCF [26] proposes a hyperbolic GCN model for CF. To alleviate the power-law distribution in recommender systems, HICF [37] aims to improve the attention of tail items in hyperbolic spaces, which makes the pull and push process geometric-aware. On the other hand, to reveal the intent factors across geometric spaces, GDCF [39] learns geometric disentangled representations associated with user intentions and various geometries. It's worth emphasizing that KBHP [28] and LKGR [5] are the most similar to our work. Both of them combine KG and hyperbolic representation learning. Different from existing methods that solely adopt a specific manifold, we introduce a unified manifold into KG-based methods to capture the structural information on global level and automatically learn the optimal curvature in the data.

## 6 CONCLUSION

In this work, we proposed a novel recommendation model named MCKG. Our key motivation is that existing non-euclidean recommendation methods have the following issues: 1) existing noneuclidean methods treat 𝜅 as a trainable parameter, but model parameters constraints depend on the changes of curvature. Once the curvature 𝜅 varies, the internal structure of manifold may be destroyed, and further unable to obtain high-quality embeddings; 2) Existing mix-curvature methods directly operate on various manifolds without considering the heterogeneity. To address the above problems, we introduce the unified space can interpolate smoothly between all geometries of constant curvature, apply multiple spaces fusion to obtain the high-quality embedding, and finally propose a geometry-aware optimization strategy compatible with hyperbolic and spherical spaces.

In future work, we will further study the mix-curvature method for other application scenarios, e.g., natural language processing [6, 19], computer vision [7, 11], such non-Euclidean modeling strategy learns embeddings of correlated data successfully with reduced distortion.

## ACKNOWLEDGMENTS

This research work is supported by the National Key Research and Development Program of China under Grant No. 2021ZD0113602, the National Natural Science Foundation of China under Grant Nos. 62176014, 62276015, 62206266, the Fundamental Research Funds for the Central Universities.

## REFERENCES

- [1] Gregor Bachmann, Gary Bécigneul, and Octavian Ganea. 2020. Constant curvature graph convolutional networks. In International Conference on Machine Learning . PMLR, 486-496.
- [2] Hongyun Cai, Vincent W Zheng, and Kevin Chen-Chuan Chang. 2018. A comprehensive survey of graph embedding: Problems, techniques, and applications. IEEE Transactions on Knowledge and Data Engineering 30, 9 (2018), 1616-1637.

- [3] Ines Chami, Zhitao Ying, Christopher Ré, and Jure Leskovec. 2019. Hyperbolic graph convolutional neural networks. Advances in neural information processing systems 32 (2019).
- [4] Lei Chen, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. 2020. Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach. In Proceedings of the AAAI conference on artificial intelligence , Vol. 34. 27-34.
- [5] Yankai Chen, Menglin Yang, Yingxue Zhang, Mengchen Zhao, Ziqiao Meng, Jianye Hao, and Irwin King. 2022. Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining . 94-102.
- [6] Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K Reddy. 2022. ANTHEM: Attentive hyperbolic entity model for product search. (2022).
- [7] Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, and Ivan Oseledets. 2022. Hyperbolic vision transformers: Combining improvements in metric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 7409-7419.
- [8] Daniele Grattarola, Daniele Zambon, Cesare Alippi, and Lorenzo Livi. 2018. Learning graph embeddings on constant-curvature manifolds for change detection in graph streams. stat 1050 (2018), 16.
- [9] Albert Gu, Frederic Sala, Beliz Gunel, and Christopher Ré. 2018. Learning mixedcurvature representations in product spaces. In International Conference on Learning Representations .
- [10] Albert Gu, Frederic Sala, Beliz Gunel, and Christopher Ré. 2019. Learning mixedcurvature representations in product spaces. In International Conference on Learning Representations .
- [11] Yunhui Guo, Xudong Wang, Yubei Chen, and Stella X Yu. 2022. Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 11-20.
- [12] William L Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems . 1025-1035.
- [13] Zhen Han, Yunpu Ma, Peng Chen, and Volker Tresp. 2020. Dyernie: Dynamic evolution of riemannian manifold embeddings for temporal knowledge graph completion. arXiv preprint arXiv:2011.03984 (2020).
- [14] Sigurdur Helgason. 1978. Differential geometry, Lie groups and symmetric spaces. Math. Surveys Monogr 83 (1978).
- [15] Binbin Hu, Chuan Shi, Wayne Xin Zhao, and Philip S Yu. 2018. Leveraging metapath based context for top-n recommendation with a neural co-attention model. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery &amp; data mining . 1531-1540.
- [16] Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, and Edward Y Chang. 2018. Improving sequential recommendation with knowledge-enhanced memory networks. In The 41st International ACM SIGIR Conference on Research &amp; Development in Information Retrieval . 505-514.
- [17] Max Kochurov, Sergey Ivanov, and Eugeny Burnaev. [n.d.]. Are Hyperbolic Representations in Graphs Created Equal? ([n. d.]).
- [18] Qi Liu, Maximilian Nickel, and Douwe Kiela. 2019. Hyperbolic graph neural networks. Advances in Neural Information Processing Systems 32 (2019).
- [19] Zheng Liu, Xiaohan Li, Zeyu You, Tao Yang, Wei Fan, and Philip Yu. 2021. Medical triage chatbot diagnosis improvement via multi-relational hyperbolic graph neural network. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1965-1969.
- [20] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. 2016. Learning convolutional neural networks for graphs. In International conference on machine learning . PMLR, 2014-2023.
- [21] Wei Peng, Tuomas Varanka, Abdelrahman Mostafa, Henglin Shi, and Guoying Zhao. 2021. Hyperbolic deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 14 (2021), 1-28.
- [22] Yanru Qu, Ting Bai, Weinan Zhang, Jianyun Nie, and Jian Tang. 2019. An end-to-end neighborhood-based interaction model for knowledge-enhanced recommendation. In Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data . 1-9.
- [23] Frederic Sala, Chris De Sa, Albert Gu, and Christopher Ré. 2018. Representation tradeoffs for hyperbolic embeddings. In International conference on machine learning . PMLR, 4460-4469.
- [24] Frederic Sala, Chris De Sa, Albert Gu, and Christopher Re. 2018. Representation Tradeoffs for Hyperbolic Embeddings. In Proceedings of the 35th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 80) , Jennifer Dy and Andreas Krause (Eds.). PMLR, 4460-4469.
- [25] Ondrej Skopek, Octavian-Eugen Ganea, and Gary Bécigneul. 2019. Mixedcurvature Variational Autoencoders. In International Conference on Learning Representations .
- [26] Jianing Sun, Zhaoyue Cheng, Saba Zuberi, Felipe Pérez, and Maksims Volkovs. 2021. Hgcf: Hyperbolic graph convolution networks for collaborative filtering. In Proceedings of the Web Conference 2021 . 593-601.
- [27] Rui Sun, Xuezhi Cao, Yan Zhao, Junchen Wan, Kun Zhou, Fuzheng Zhang, Zhongyuan Wang, and Kai Zheng. 2020. Multi-modal knowledge graphs for recommender systems. In Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management . 1405-1414.
- [28] Chang-You Tai, Chien-Kun Huang, Liang-Ying Huang, and Lun-Wei Ku. 2021. Knowledge Based Hyperbolic Propagation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1945-1949.
- [29] Lucas Vinh Tran, Yi Tay, Shuai Zhang, Gao Cong, and Xiaoli Li. 2020. Hyperml: A boosting metric learning approach in hyperbolic space for recommender systems. In Proceedings of the 13th International Conference on Web Search and Data Mining . 609-617.
- [30] Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, and Minyi Guo. 2018. Ripplenet: Propagating user preferences on the knowledge graph for recommender systems. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management . 417-426.
- [31] Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. DKN: Deep knowledge-aware network for news recommendation. In Proceedings of the 2018 world wide web conference . 1835-1844.
- [32] Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, and Minyi Guo. 2019. Knowledge graph convolutional networks for recommender systems. In The world wide web conference . 3307-3313.
- [33] Shen Wang, Xiaokai Wei, Cicero Nogueira Nogueira dos Santos, Zhiguo Wang, Ramesh Nallapati, Andrew Arnold, Bing Xiang, Philip S Yu, and Isabel F Cruz. 2021. Mixed-curvature multi-relational graph neural network for knowledge graph completion. In Proceedings of the Web Conference 2021 . 1761-1771.
- [34] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. Kgat: Knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining . 950-958.
- [35] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval . 165-174.
- [36] Xiang Wang, Tinglin Huang, Dingxian Wang, Yancheng Yuan, Zhenguang Liu, Xiangnan He, and Tat-Seng Chua. 2021. Learning intents behind interactions with knowledge graph for recommendation. In Proceedings of the Web Conference 2021 . 878-887.
- [37] Menglin Yang, Zhihao Li, Min Zhou, Jiahong Liu, and Irwin King. 2022. Hicf: Hyperbolic informative collaborative filtering. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 2212-2221.
- [38] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, and Wei-Ying Ma. 2016. Collaborative knowledge base embedding for recommender systems. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining . 353-362.
- [39] Yiding Zhang, Chaozhuo Li, Xing Xie, Xiao Wang, Chuan Shi, Yuming Liu, Hao Sun, Liangjie Zhang, Weiwei Deng, and Qi Zhang. 2022. Geometric Disentangled Collaborative Filtering. (2022).
- [40] Yiding Zhang, Chaozhuo Li, Xing Xie, Xiao Wang, Chuan Shi, Yuming Liu, Hao Sun, Liangjie Zhang, Weiwei Deng, and Qi Zhang. 2022. Geometric Disentangled Collaborative Filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 80-90.