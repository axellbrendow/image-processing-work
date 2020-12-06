# Trabalho de Processamento de Imagens

Repositório para o trabalho da disciplina: Processamento de Imagens

## Descrição

A densidade da mama é comprovadamente relacionada com o risco do desenvolvimento de câncer, uma vez que mulheres com uma maior densidade mamária podem esconder lesões, levando o câncer a ser detectado tardiamente. A escala de densidade chamada BIRADS foi desenvolvida pelo American College of Radiology e informa os radiologistas sobre a diminuição da sensibilidade do exame com o aumento da densidade da mama. BI-RADS definem a densidade como sendo quase inteiramente composta por gordura (densidade I), por tecido fibrobroglandular difuso (densidade II), por tecido denso heterogêneo (III) e por tecido extremamente denso (IV). A mamografia é a principal ferramenta de rastreio do câncer e radiologistas avaliam a densidade da mama com base na análise visual das imagens.

## Motivação e objetivos do trabalho

Implementar um programa que leia imagens de exames mamográficos e possibilite diferenciar a densidade das mamas no intuito de identificar tumores ou outros problemas.

## Descrição das técnicas implementadas para a solução, principalmente do classificador

O classificador escolhido para o trabalho é uma rede neural completamente conectada.

Tivemos 71% de acurácia com 2000 épocas de treino, usando 256 neurônios na primeira camada da rede neural e usando a função de ativação [retificadora](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

A segunda e última camada da rede neural tem 4 neurônios que correspondem às 4 classes BIRADS.

O otimizador da rede neural usa o algoritmo de Adam e a métrica extraída é a acurácia.

As imagens podem ser descritas por todos os 13 primeiros descritores do artigo "[Textural Features for Image Classification](https://ieeexplore.ieee.org/document/4309314)" mais os 7 momentos invariantes de Hu:

- Energia
- Contraste
- Correlação
- Variância
- Homogeneidade
- Soma média
- Soma da variância
- Soma da entropia
- Entropia
- Diferença da variância
- Diferença da entropia
- Medida de informação de correlação 12
- Medida de informação de correlação 13
- Todos os 7 momentos invariantes de Hu

Formato dos dados de treinamento e teste:

A rede neural recebe um vetor D (descritores) onde D\[i\] contém o arranjo de descritores de uma imagem e um vetor C (classes) onde C\[i\] contém a classe esperada para essa imagem.

## Como usar ?

- Esse projeto só funciona com a existência da pasta `imagens` disponibilizada pelo professor da disciplina. Não pude trazer essa pasta para o repositório pois as imagens não são de livre acesso.

Windows:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Linux:

```
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 main.py
```

## Referências

**Move and zoom a tkinter canvas with mouse**. stackoverflow, 2020. Disponível em: https://stackoverflow.com/questions/25787523/move-and-zoom-a-tkinter-canvas-with-mouse/48069295#48069295. Acesso em: 20, Novembro de 2020.

**Tkinter canvas zoom move pan**. stackoverflow, 2020. Disponível em: https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan/48137257#48137257. Acesso em: 20, Novembro de 2020.

**Textural Features for Image Classification**. ieeexplore, 2020. Disponível em: https://ieeexplore.ieee.org/document/4309314. Acesso em: 20, Novembro de 2020.

**How-can-i-convert-an-rgb-image-into-grayscale-in-python**. stackoverflow, 2020. Disponível em: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python. Acesso em: 20, Novembro de 2020.

**Opencv-shape-descriptor-hu-moments-example**. pyimagesearch, 2020. Disponível em: https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/. Acesso em: 20, Novembro de 2020.

**Texture Recognition using Haralick Texture and Python**. gogul, 2020. Disponível em: https://gogul.dev/software/texture-recognition. Acesso em: 20, Novembro de 2020.

**Tensorflow Image Classification | Build Your Own Image Classifier In Tensorflow | Edureka**. youtube, 2020. Disponível em: https://www.youtube.com/watch?v=AACPaoDsd50&ab_channel=edureka%21. Acesso em: 20, Novembro de 2020.

**Treine sua primeira rede neural: classificação básica**. tensorflow, 2020. Disponível em: https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br#fa%C3%A7a_predi%C3%A7%C3%B5es. Acesso em: 20, Novembro de 2020.

**Confusion Matrix for Your Multi-Class Machine Learning Model**. towardsdatascience, 2020. Disponível em: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826. Acesso em: 20, Novembro de 2020.

**Scikit-learn: How to obtain True Positive, True Negative, False Positive and False Negative**. stackoverflow, 2020. Disponível em: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal. Acesso em: 20, Novembro de 2020.
