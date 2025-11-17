
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
annotations_creators:
- crowdsourced
language_creators:
- crowdsourced
language:
- en
license:
- apache-2.0
task_categories:
- text-generation
pretty_name: task1288_glue_mrpc_paraphrasing
dataset_info:
  config_name: plain_text
  features:
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: train
    num_examples: 3256
  - name: valid
    num_examples: 407
  - name: test
    num_examples: 408
---

# Dataset Card for Natural Instructions (https://github.com/allenai/natural-instructions) Task: task1288_glue_mrpc_paraphrasing

## Dataset Description

- **Homepage:** https://github.com/allenai/natural-instructions
- **Paper:** https://arxiv.org/abs/2204.07705
- **Paper:** https://arxiv.org/abs/2407.00066
- **Point of Contact:** [Rickard Brüel Gabrielsson](mailto:brg@mit.edu)

## Additional Information

### Citation Information

The following paper introduces the corpus in detail. If you use the corpus in published work, please cite it: 
```bibtex
@misc{wang2022supernaturalinstructionsgeneralizationdeclarativeinstructions,
    title={Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks}, 
    author={Yizhong Wang and Swaroop Mishra and Pegah Alipoormolabashi and Yeganeh Kordi and Amirreza Mirzaei and Anjana Arunkumar and Arjun Ashok and Arut Selvan Dhanasekaran and Atharva Naik and David Stap and Eshaan Pathak and Giannis Karamanolakis and Haizhi Gary Lai and Ishan Purohit and Ishani Mondal and Jacob Anderson and Kirby Kuznia and Krima Doshi and Maitreya Patel and Kuntal Kumar Pal and Mehrad Moradshahi and Mihir Parmar and Mirali Purohit and Neeraj Varshney and Phani Rohitha Kaza and Pulkit Verma and Ravsehaj Singh Puri and Rushang Karia and Shailaja Keyur Sampat and Savan Doshi and Siddhartha Mishra and Sujan Reddy and Sumanta Patro and Tanay Dixit and Xudong Shen and Chitta Baral and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi and Daniel Khashabi},
    year={2022},
    eprint={2204.07705},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2204.07705}, 
}
```

More details can also be found in the following paper:
```bibtex
@misc{brüelgabrielsson2024compressserveservingthousands,
    title={Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead}, 
    author={Rickard Brüel-Gabrielsson and Jiacheng Zhu and Onkar Bhardwaj and Leshem Choshen and Kristjan Greenewald and Mikhail Yurochkin and Justin Solomon},
    year={2024},
    eprint={2407.00066},
    archivePrefix={arXiv},
    primaryClass={cs.DC},
    url={https://arxiv.org/abs/2407.00066}, 
}
```

### Contact Information

For any comments or questions, please email [Rickard Brüel Gabrielsson](mailto:brg@mit.edu)

Output:
{
    "extracted_code": ""
}
