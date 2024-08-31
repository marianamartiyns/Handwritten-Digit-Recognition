# 🖋️ **Reconhecimento de Dígitos Escritos à Mão**

<a href="https://github.com/flipfelly"><img height="25" src="https://img.shields.io/badge/-Felipe Gontijo-black?logo=github&style=flat-square"/></a> 
<a href="https://github.com/marianamartiyns"><img height="25" src="https://img.shields.io/badge/-Mariana Martins-black?logo=github&style=flat-square"/></a>

### 🤖 Projeto de Aprendizagem de Máquina

Bem-vindo ao projeto **Reconhecimento de Dígitos**, onde o objetivo principal é desenvolver e implementar três classificadores para identificar dígitos escritos à mão, utilizando o dataset adaptado do **MNIST**, que contém imagens em escala de cinza dos dígitos **0, 1, 4 e 5**. Este é um problema clássico na área de visão computacional, essencial para a aplicação de técnicas de aprendizado de máquina.

### 📂 **Descrição do Projeto**

1. **Dataset**:  
   O dataset MNIST adaptado é composto por imagens de **28x28 pixels** (totalizando **784 pixels** por imagem). Cada pixel possui um valor que indica seu tom de cinza, variando de **0 (branco)** a **255 (preto)**. Os dados de entrada estão organizados em arquivos CSV, onde a primeira coluna indica o dígito correspondente, e as demais colunas contêm os valores dos pixels.

2. **Classificadores**:  
   Foram implementados três modelos lineares de aprendizado de máquina:
   - **Perceptron**
   - **Regressão Linear**
   - **Regressão Logística**

3. **Redução de Dimensão**:  
   Para melhorar a eficiência dos modelos, foi realizada uma redução na dimensionalidade dos dados, sintetizando as informações em apenas duas características principais:
   - **Intensidade da Imagem**
   - **Simetria**

4. **Avaliação**:  
   O desempenho dos classificadores será avaliado com base na precisão na identificação dos dígitos, permitindo comparações entre as diferentes abordagens.

<img align="right" width ='40px' src ='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg'> </a>
<img align="right" width ='40px' src ='https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg'> </a>

