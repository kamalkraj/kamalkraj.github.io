/*Research Details Table*/

const researchTable = document.querySelector(".main");

const research = [
    {
        title : "Small-Bench NLP: Benchmark for small single GPU trained models in Natural Language Processing",
        authors : "Kamal Raj Kanakarajan, Bhuvana Kundumani, Malaikannan Sankarasubbu",
        conferences : "arxiv preprint",
        researchYr : 2021,
        citebox : "popup1",
        image : "assets/images/research-page/smallbenchnlp.png",
        citation: {
            vancouver: "Kanakarajan, Kamal Raj, et al. “Small-Bench NLP: Benchmark for Small Single GPU Trained Models in Natural Language Processing.” ArXiv:2109.10847 [Cs], Sept. 2021. arXiv.org, http://arxiv.org/abs/2109.10847."
        },
        abstract: "Recent progress in the Natural Language Processing domain has given us several State-of-the-Art (SOTA) pretrained models which can be finetuned for specific tasks. These large models with billions of parameters trained on numerous GPUs/TPUs over weeks are leading in the benchmark leaderboards. In this paper, we discuss the need for a benchmark for cost and time effective smaller models trained on a single GPU. This will enable researchers with resource constraints experiment with novel and innovative ideas on tokenization, pretraining tasks, architecture, fine tuning methods etc. We set up Small-Bench NLP, a benchmark for small efficient neural language models trained on a single GPU. Small-Bench NLP benchmark comprises of eight NLP tasks on the publicly available GLUE datasets and a leaderboard to track the progress of the community. Our ELECTRA-DeBERTa (15M parameters) small model architecture achieves an average score of 81.53 which is comparable to that of BERT-Base's 82.20 (110M parameters). Our models, code and leaderboard are available at https://github.com/smallbenchnlp",
        absbox: "absPopup1"
    },

    {
        title : "BioELECTRA:Pretrained Biomedical text Encoder using Discriminators",
        authors : "Kamal raj Kanakarajan, Bhuvana Kundumani, Malaikannan Sankarasubbu",
        conferences : "North American Chapter of the Association for Computational Linguistics",
        researchYr : 2021,
        citebox : "popup2",
        image : "assets/images/research-page/bioelectra.png",
        citation: {
            vancouver: "Kanakarajan, Kamal raj, et al. “BioELECTRA:Pretrained Biomedical Text Encoder Using Discriminators.” Proceedings of the 20th Workshop on Biomedical Language Processing, Association for Computational Linguistics, 2021, pp. 143–54. DOI.org (Crossref), https://doi.org/10.18653/v1/2021.bionlp-1.16."
        },
        abstract: "Recent advancements in pretraining strategies in NLP have shown a significant improvement in the performance of models on various text mining tasks. We apply ‘replaced token detection’ pretraining technique proposed by ELECTRA and pretrain a biomedical language model from scratch using biomedical text and vocabulary. We introduce BioELECTRA, a biomedical domain-specific language encoder model that adapts ELECTRA for the Biomedical domain. WE evaluate our model on the BLURB and BLUE biomedical NLP benchmarks. BioELECTRA outperforms the previous models and achieves state of the art (SOTA) on all the 13 datasets in BLURB benchmark and on all the 4 Clinical datasets from BLUE Benchmark across 7 different NLP tasks. BioELECTRA pretrained on PubMed and PMC full text articles performs very well on Clinical datasets as well. BioELECTRA achieves new SOTA 86.34%(1.39% accuracy improvement) on MedNLI and 64% (2.98% accuracy improvement) on PubMedQA dataset.",
        absbox: "absPopup2"
    },

    {
        title : "Saama Research at MEDIQA 2019: Pre-trained BioBERT with Attention Visualisation for Medical Natural Language Inference",
        authors : "Kamal raj Kanakarajan, Suriyadeepan Ramamoorthy, Vaidheeswaran Archana, Soham Chatterjee, Malaikannan Sankarasubbu",
        conferences : "Annual Meeting of the Association for Computational Linguistics (ACL)",
        researchYr : 2019,
        citebox : "popup3",
        image : "assets/images/research-page/mediqa.png",
        citation: {
            vancouver: "Kanakarajan, Kamal raj, et al. “Saama Research at MEDIQA 2019: Pre-Trained BioBERT with Attention Visualisation for Medical Natural Language Inference.” Proceedings of the 18th BioNLP Workshop and Shared Task, Association for Computational Linguistics, 2019, pp. 510–16. DOI.org (Crossref), https://doi.org/10.18653/v1/W19-5055."
        },
        abstract: "Natural Language inference is the task of identifying relation between two sentences as entailment, contradiction or neutrality. MedNLI is a biomedical flavour of NLI for clinical domain. This paper explores the use of Bidirectional Encoder Representation from Transformer (BERT) for solving MedNLI. The proposed model, BERT pre-trained on PMC, PubMed and fine-tuned on MIMICIII v1.4, achieves state of the art results on MedNLI (83.45%) and an accuracy of 78.5% in MEDIQA challenge. The authors present an analysis of the attention patterns that emerged as a result of training BERT on MedNLI using a visualization tool, bertviz.",
        absbox: "absPopup3"
    },

    {
        title : "PHI Scrubber: A Deep Learning Approach",
        authors : "Abhai Kollara Dilip, Kamal Raj K, Malaikannan Sankarasubbu",
        conferences : "arxiv preprint",
        researchYr : 2018,
        citebox : "popup4",
        image : "assets/images/research-page/phiscrubber.png",
        citation: {
            vancouver: "Dilip, Abhai Kollara, et al. “PHI Scrubber: A Deep Learning Approach.” ArXiv:1808.01128 [Cs, Stat], Aug. 2018. arXiv.org, http://arxiv.org/abs/1808.01128."
        },
        abstract: "Confidentiality of patient information is an essential part of Electronic Health Record System. Patient information, if exposed, can cause a serious damage to the privacy of individuals receiving healthcare. Hence it is important to remove such details from physician notes. A system is proposed which consists of a deep learning model where a de-convolutional neural network and bi-directional LSTM-CNN is used along with regular expressions to recognize and eliminate the individually identifiable information. This information is then removed from a medical practitioner's data which further allows the fair usage of such information among researchers and in clinical trials.",
        absbox: "absPopup4"
    },

    // {
    //     title : "Dual Super-Resolution Learning for Semantic Segmentation",
    //     authors : "Wang, Li and Li, Dong and Zhu, Yousong and Tian, Lu and Shan, Yi",
    //     conferences : "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    //     researchYr : 2020,
    //     citebox : "popup5",
    //     image : "assets/images/research-page/semanticSegmentation.png",
    //     citation: {
    //         vancouver: "Wang, Li and Li, Dong and Zhu, Yousong and Tian, Lu and Shan, Yi. Dual Super-Resolution Learning for Semantic Segmentation. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020."
    //     },
    //     abstract: "This is currently left empty and this can be considered as a dummy data 5",
    //     absbox: "absPopup5"
    // },

    // {
    //     title : "Deep Unfolding Network for Image Super-Resolution",
    //     authors : "Zhang, Kai and Van Gool, Luc and Timofte, Radu",
    //     conferences : "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    //     researchYr : 2020,
    //     citebox : "popup6",
    //     image : "assets/images/research-page/deepNetwork.png",
    //     citation: {
    //         vancouver: "Zhang, Kai and Van Gool, Luc and Timofte, Radu. Deep Unfolding Network for Image Super-Resolution. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020."
    //     },
    //     abstract: "This is currently left empty and this can be considered as a dummy data 6",
    //     absbox: "absPopup6"
    // },

    // {
    //     title : "Unsupervised Learning for Intrinsic Image Decomposition From a Single Image",
    //     authors : "Liu, Yunfei and Li, Yu and You, Shaodi and Lu, Feng",
    //     conferences : "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    //     researchYr : 2020,
    //     citebox : "popup7",
    //     image : "assets/images/research-page/imageDecomposition.png",
    //     citation: {
    //         vancouver: "Liu, Yunfei and Li, Yu and You, Shaodi and Lu, Feng. Unsupervised Learning for Intrinsic Image Decomposition From a Single Image. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020."
    //     },
    //     abstract: "This is currently left empty and this can be considered as a dummy data 7",
    //     absbox: "absPopup7"
    // },
    // {
    //     title : "Forward and Backward Information Retention for Accurate Binary Neural Networks",
    //     authors : "Qin, Haotong and Gong, Ruihao and Liu, Xianglong and Shen, Mingzhu and Wei, Ziran and Yu, Fengwei and Song, Jingkuan",
    //     conferences : "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    //     researchYr : 2020,
    //     citebox : "popup8",
    //     image : "assets/images/research-page/neuralNetworks.jpg",
    //     citation: {
    //         vancouver: "Qin, Haotong and Gong, Ruihao and Liu, Xianglong and Shen, Mingzhu and Wei, Ziran and Yu, Fengwei and Song, Jingkuan. Forward and Backward Information Retention for Accurate Binary Neural Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020."
    //     },
    //     abstract: "This is currently left empty and this can be considered as a dummy data 8",
    //     absbox: "absPopup8"
    // }
];
AOS.init();   
const fillData = () => {
    let output = "";
    research.forEach(
        ({image, title, authors, conferences, researchYr, citebox, citation, absbox, abstract}) =>
        (output +=`
            <tr data-aos="zoom-in-left"> 
                <td class="imgCol"><img src="${image}" class="rImg"></td>
                <td class = "researchTitleName">
                    <div>
                        <span class="imgResponsive">
                            <img src="${image}" class="imgRes">
                        </span>
                    </div>
                    <a href="#0" class="paperTitle"> ${title} </a> 
                    <div> ${authors} </div> <div class="rConferences"> ${conferences} 
                        <div class="researchY">${researchYr}</div>
                    </div>
        
                    <!--CITE BUTTON-->
                    <div class="d-flex" style="margin-right:5%;">
                        <button class="button button-accent button-small text-right button-abstract " type="button" data-toggle="collapse" data-target="#${absbox}" aria-expanded="false" aria-controls="${absbox}">
                            ABSTRACT
                        </button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                
                        <button class="button button-accent button-small text-right button-abstract " type="button" data-toggle="collapse" data-target="#${citebox}" aria-expanded="false" aria-controls="${citebox}">
                            CITE
                        </button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    </div>
                    <div id="${absbox}" class="collapse" aria-labelledby="headingTwo" data-parent=".collapse">
                        <div class="card-body">
                            ${abstract}    
                        </div>
                    </div>
                    <div id="${citebox}" class="collapse" aria-labelledby="headingTwo" data-parent=".collapse">
                        <div class="card-body">
                            ${citation.vancouver}    
                        </div>
                    </div>
                </td>
            </tr>`)
        );
    researchTable.innerHTML = output;

};
document.addEventListener("DOMContentLoaded", fillData);

