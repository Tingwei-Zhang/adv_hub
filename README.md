<h1 align="center"> 
Adversarial Hubness in Multi-Modal Retrieval </h1>

<p align="center"> <i>Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov</i></p>

Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to
many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly)
appear relevant to many queries.

In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts.

We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated with respect to 100 randomly selected target queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

Paper link:
[https://arxiv.org/pdf/2412.14113](https://arxiv.org/pdf/2412.14113)

<img src="image/embedding_plot_new.png" alt="drawing" width="600"/>


Most experiments run on a single NVIDIA A40 40g GPU.

Please feel free to email: tingwei@cs.cornell.edu or raise an issue.
