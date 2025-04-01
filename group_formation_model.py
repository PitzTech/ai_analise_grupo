import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import os
import datetime

class AlgoritmoGeneticoAVA:
    """
    Implementação de Algoritmo Genético para formação de grupos em Ambientes Virtuais de Aprendizagem (AVA)
    baseado no artigo "Uma Abordagem Baseada em Algoritmo Genético para Formação de Grupos de Estudos
    em Ambientes Virtuais de Aprendizagem"
    """

    def __init__(self, caracteristicas_alunos, num_grupos, tam_populacao=100,
                 pc=0.1, pm=0.03, num_geracoes=100):
        """
        Inicializa o AG com os parâmetros definidos

        Parâmetros:
        caracteristicas_alunos (numpy.ndarray): Matriz com características dos alunos normalizadas
        num_grupos (int): Número de grupos a serem formados
        tam_populacao (int): Tamanho da população
        pc (float): Probabilidade de cruzamento (recomendado: 0.1)
        pm (float): Probabilidade de mutação (recomendado: 0.03)
        num_geracoes (int): Número de gerações
        """
        self.X = caracteristicas_alunos
        self.num_alunos = caracteristicas_alunos.shape[0]
        self.num_caracteristicas = caracteristicas_alunos.shape[1]
        self.num_grupos = num_grupos
        self.tam_populacao = tam_populacao
        self.pc = pc
        self.pm = pm
        self.num_geracoes = num_geracoes
        self.historico_fitness = []

    def inicializar_populacao(self):
        """
        Inicializa a população com soluções aleatórias

        Retorna:
        numpy.ndarray: População inicial
        """
        # Cada indivíduo é um vetor onde cada posição representa um aluno
        # e o valor em cada posição representa o grupo ao qual o aluno pertence
        populacao = np.random.randint(1, self.num_grupos + 1,
                                      size=(self.tam_populacao, self.num_alunos))
        return populacao

    def calcular_centroides(self, individuo):
        """
        Calcula os centroides dos grupos formados pelo indivíduo

        Parâmetros:
        individuo (numpy.ndarray): Vetor que representa a solução

        Retorna:
        dict: Dicionário com os centroides de cada grupo
        """
        centroides = {}

        # Para cada grupo
        for g in range(1, self.num_grupos + 1):
            # Encontra os alunos que pertencem ao grupo g
            indices_alunos = np.where(individuo == g)[0]

            if len(indices_alunos) > 0:
                # Calcula o centroide como a média das características dos alunos do grupo
                centroide = np.mean(self.X[indices_alunos], axis=0)
                centroides[g] = centroide
            else:
                # Se não houver alunos no grupo, cria um centroide aleatório
                centroides[g] = np.random.random(self.num_caracteristicas)

        return centroides

    def calcular_fitness(self, individuo):
        """
        Calcula o fitness de um indivíduo conforme descrito no artigo

        Parâmetros:
        individuo (numpy.ndarray): Vetor que representa a solução

        Retorna:
        float: Valor de fitness do indivíduo
        """
        # Calcula os centroides dos grupos
        centroides = self.calcular_centroides(individuo)

        # Calcular a soma das distâncias euclidianas dos alunos aos centroides de seus grupos
        soma_distancias = 0
        for i in range(self.num_alunos):
            grupo_aluno = individuo[i]
            centroide_grupo = centroides[grupo_aluno]

            # Calcula a distância euclidiana entre o aluno e o centroide
            distancia = np.linalg.norm(self.X[i] - centroide_grupo)
            soma_distancias += distancia

        # O fitness é o inverso da soma das distâncias (multiplicado por um fator de escala)
        # Quanto menor a soma das distâncias, maior o fitness
        fator_escala = 100
        fitness = fator_escala / (1 + soma_distancias)

        return fitness

    def selecao_roleta(self, fitness_populacao):
        """
        Implementa o método de seleção por roleta (roulette wheel)

        Parâmetros:
        fitness_populacao (numpy.ndarray): Valores de fitness da população

        Retorna:
        list: Índices dos indivíduos selecionados
        """
        # Calcula a soma total de fitness
        soma_fitness = np.sum(fitness_populacao)

        if soma_fitness == 0:
            # Se a soma for zero, seleciona uniformemente
            prob_selecao = np.ones(self.tam_populacao) / self.tam_populacao
        else:
            # Calcula a probabilidade de seleção proporcional ao fitness
            prob_selecao = fitness_populacao / soma_fitness

        # Seleciona indivíduos usando roleta
        selecionados = np.random.choice(
            self.tam_populacao,
            size=self.tam_populacao,
            p=prob_selecao,
            replace=True
        )

        return selecionados

    def cruzamento_uniforme(self, populacao, indices_selecionados):
        """
        Implementa o cruzamento uniforme como descrito no artigo

        Parâmetros:
        populacao (numpy.ndarray): População atual
        indices_selecionados (list): Índices dos indivíduos selecionados

        Retorna:
        numpy.ndarray: Nova população após cruzamento
        """
        nova_populacao = np.zeros((self.tam_populacao, self.num_alunos), dtype=int)

        # Mantém o melhor indivíduo (estratégia elitista)
        melhor_indice = np.argmax([self.calcular_fitness(populacao[i]) for i in indices_selecionados])
        nova_populacao[0] = populacao[indices_selecionados[melhor_indice]]

        # Realiza o cruzamento para o restante da população
        for i in range(1, self.tam_populacao, 2):
            # Seleciona dois pais da população
            pai1_idx = indices_selecionados[random.randint(0, len(indices_selecionados)-1)]
            pai2_idx = indices_selecionados[random.randint(0, len(indices_selecionados)-1)]

            pai1 = populacao[pai1_idx]
            pai2 = populacao[pai2_idx]

            # Decide se ocorrerá cruzamento
            if random.random() < self.pc:
                # Cria máscara de cruzamento uniforme
                mascara = np.random.randint(0, 2, size=self.num_alunos)

                # Cria filhos usando a máscara
                filho1 = np.where(mascara == 0, pai1, pai2)
                filho2 = np.where(mascara == 0, pai2, pai1)
            else:
                # Se não ocorrer cruzamento, os filhos são cópias dos pais
                filho1 = pai1.copy()
                filho2 = pai2.copy()

            # Adiciona os filhos à nova população
            if i < self.tam_populacao:
                nova_populacao[i] = filho1
            if i+1 < self.tam_populacao:
                nova_populacao[i+1] = filho2

        return nova_populacao

    def mutacao_gene(self, populacao):
        """
        Implementa a mutação de gene como descrito no artigo

        Parâmetros:
        populacao (numpy.ndarray): População atual

        Retorna:
        numpy.ndarray: População após mutação
        """
        # Cria uma cópia da população
        nova_populacao = populacao.copy()

        # Para cada gene em cada cromossomo (exceto o primeiro - elitismo)
        for i in range(1, self.tam_populacao):
            for j in range(self.num_alunos):
                # Decide se ocorrerá mutação
                if random.random() < self.pm:
                    # Lista de possíveis valores (grupos) excluindo o valor atual
                    valores_possiveis = list(range(1, self.num_grupos + 1))
                    valores_possiveis.remove(nova_populacao[i, j])

                    # Atribui um novo valor aleatório dentre os possíveis
                    nova_populacao[i, j] = random.choice(valores_possiveis)

        return nova_populacao

    def executar(self, arquivo_saida=None):
        """
        Executa o algoritmo genético

        Parâmetros:
        arquivo_saida (file): Arquivo para salvar o progresso (opcional)

        Retorna:
        tuple: Melhor solução encontrada e seu fitness
        """
        # Inicializa a população
        populacao = self.inicializar_populacao()

        if arquivo_saida:
            arquivo_saida.write("### Execução do Algoritmo Genético ###\n")
            arquivo_saida.write(f"Número de alunos: {self.num_alunos}\n")
            arquivo_saida.write(f"Número de grupos: {self.num_grupos}\n")
            arquivo_saida.write(f"Tamanho da população: {self.tam_populacao}\n")
            arquivo_saida.write(f"Probabilidade de cruzamento: {self.pc}\n")
            arquivo_saida.write(f"Probabilidade de mutação: {self.pm}\n")
            arquivo_saida.write(f"Número de gerações: {self.num_geracoes}\n\n")
            arquivo_saida.write("Progresso por geração:\n")

        # Para cada geração com barra de progresso
        with tqdm(total=self.num_geracoes, desc="Executando AG") as pbar:
            for geracao in range(self.num_geracoes):
                # Calcula o fitness de cada indivíduo
                fitness_populacao = np.array([self.calcular_fitness(ind) for ind in populacao])

                # Guarda o melhor fitness da geração
                melhor_fitness = np.max(fitness_populacao)
                self.historico_fitness.append(melhor_fitness)

                # Seleciona indivíduos para reprodução
                indices_selecionados = self.selecao_roleta(fitness_populacao)

                # Aplica cruzamento
                populacao = self.cruzamento_uniforme(populacao, indices_selecionados)

                # Aplica mutação
                populacao = self.mutacao_gene(populacao)

                # Escreve progresso no arquivo se fornecido
                if arquivo_saida and (geracao % 10 == 0 or geracao == self.num_geracoes - 1):
                    arquivo_saida.write(f"Geração {geracao + 1}: Melhor fitness = {melhor_fitness:.6f}\n")

                # Atualiza a barra de progresso
                pbar.update(1)
                pbar.set_postfix({"Melhor Fitness": f"{melhor_fitness:.4f}"})

        # Calcula o fitness da população final
        fitness_final = np.array([self.calcular_fitness(ind) for ind in populacao])

        # Retorna o melhor indivíduo e seu fitness
        melhor_indice = np.argmax(fitness_final)
        return populacao[melhor_indice], fitness_final[melhor_indice]

    def plotar_evolucao(self, arquivo_saida=None, salvar_figura=True):
        """
        Plota a evolução do fitness ao longo das gerações

        Parâmetros:
        arquivo_saida (file): Arquivo para salvar o caminho da figura (opcional)
        salvar_figura (bool): Se True, salva a figura em um arquivo
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.historico_fitness)), self.historico_fitness)
        plt.title('Evolução do Fitness')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)

        if salvar_figura:
            # Cria diretório para figuras se não existir
            if not os.path.exists('figuras'):
                os.makedirs('figuras')

            # Gera nome de arquivo com timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_figura = f"figuras/evolucao_fitness_{timestamp}.png"
            plt.savefig(caminho_figura)

            if arquivo_saida:
                arquivo_saida.write(f"\nGráfico de evolução salvo em: {caminho_figura}\n")

        # plt.show()


def normalizar_dados(dados):
    """
    Normaliza os dados entre 0 e 1 como descrito no artigo

    Parâmetros:
    dados (numpy.ndarray): Matriz de dados original

    Retorna:
    numpy.ndarray: Matriz de dados normalizada
    """
    num_alunos, num_atributos = dados.shape
    dados_normalizados = np.zeros_like(dados, dtype=float)

    for j in range(num_atributos):
        valor_min = np.min(dados[:, j])
        valor_max = np.max(dados[:, j])

        if valor_max == valor_min:
            dados_normalizados[:, j] = 0.5  # Evita divisão por zero
        else:
            dados_normalizados[:, j] = (dados[:, j] - valor_min) / (valor_max - valor_min)

    return dados_normalizados


def formacao_grupos_aleatoria(num_alunos, num_grupos, dados_alunos):
    """
    Implementa formação de grupos aleatória para comparação

    Parâmetros:
    num_alunos (int): Número de alunos
    num_grupos (int): Número de grupos a formar
    dados_alunos (numpy.ndarray): Dados dos alunos

    Retorna:
    tuple: Grupos formados e a soma das distâncias
    """
    # Cria solução aleatória
    solucao = np.random.randint(1, num_grupos + 1, size=num_alunos)

    # Calcula centroides
    centroides = {}
    for g in range(1, num_grupos + 1):
        indices_alunos = np.where(solucao == g)[0]
        if len(indices_alunos) > 0:
            centroides[g] = np.mean(dados_alunos[indices_alunos], axis=0)
        else:
            centroides[g] = np.random.random(dados_alunos.shape[1])

    # Calcula a soma das distâncias
    soma_distancias = 0
    for i in range(num_alunos):
        grupo_aluno = solucao[i]
        centroide_grupo = centroides[grupo_aluno]
        distancia = np.linalg.norm(dados_alunos[i] - centroide_grupo)
        soma_distancias += distancia

    return solucao, soma_distancias


# Função para comparar os métodos
def comparar_metodos(dados_alunos, num_grupos, num_instancias=30, arquivo_saida=None):
    """
    Compara o método proposto com o método aleatório

    Parâmetros:
    dados_alunos (numpy.ndarray): Dados dos alunos
    num_grupos (int): Número de grupos a formar
    num_instancias (int): Número de instâncias de simulação
    arquivo_saida (file): Arquivo para salvar os resultados (opcional)

    Retorna:
    tuple: Média e intervalo de confiança para cada método
    """
    num_alunos = dados_alunos.shape[0]

    # Normaliza os dados
    dados_normalizados = normalizar_dados(dados_alunos)

    resultados_ag = []
    resultados_aleatorio = []

    if arquivo_saida:
        arquivo_saida.write("\n### Comparação de Métodos ###\n")
        arquivo_saida.write(f"Número de instâncias: {num_instancias}\n\n")
        arquivo_saida.write("Resultados por instância:\n")
        arquivo_saida.write("Instância | Distância AG | Distância Aleatório\n")
        arquivo_saida.write("-" * 50 + "\n")

    # Adicionando barra de progresso para a comparação
    with tqdm(total=num_instancias, desc="Comparando métodos") as pbar:
        for i in range(num_instancias):
            # Executa o método proposto (AG)
            ag = AlgoritmoGeneticoAVA(
                dados_normalizados,
                num_grupos,
                tam_populacao=100,
                pc=0.1,
                pm=0.03,
                num_geracoes=100
            )
            _, fitness_ag = ag.executar()

            # Converter fitness para soma de distâncias
            soma_distancias_ag = 100/fitness_ag - 1
            resultados_ag.append(soma_distancias_ag)

            # Executa o método aleatório
            _, soma_distancias_aleatorio = formacao_grupos_aleatoria(
                num_alunos,
                num_grupos,
                dados_normalizados
            )
            resultados_aleatorio.append(soma_distancias_aleatorio)

            # Salva resultados da instância atual no arquivo
            if arquivo_saida:
                arquivo_saida.write(f"{i+1:9d} | {soma_distancias_ag:12.4f} | {soma_distancias_aleatorio:19.4f}\n")

            # Atualiza a barra de progresso
            pbar.update(1)
            pbar.set_postfix({
                "AG": f"{soma_distancias_ag:.4f}",
                "Aleatório": f"{soma_distancias_aleatorio:.4f}"
            })

    # Calcula média e desvio padrão
    media_ag = np.mean(resultados_ag)
    media_aleatorio = np.mean(resultados_aleatorio)
    std_ag = np.std(resultados_ag)
    std_aleatorio = np.std(resultados_aleatorio)

    # Calcula melhoria percentual
    melhoria_percentual = ((media_aleatorio - media_ag) / media_aleatorio) * 100

    if arquivo_saida:
        arquivo_saida.write("\nResultados Finais:\n")
        arquivo_saida.write(f"Média das distâncias (AG): {media_ag:.4f} ± {std_ag:.4f}\n")
        arquivo_saida.write(f"Média das distâncias (Aleatório): {media_aleatorio:.4f} ± {std_aleatorio:.4f}\n")
        arquivo_saida.write(f"Melhoria: {melhoria_percentual:.2f}%\n")

    return media_ag, media_aleatorio


# Exemplo de uso
if __name__ == "__main__":
    # Cria diretório para resultados se não existir
    if not os.path.exists('resultados'):
        os.makedirs('resultados')

    # Gera nome de arquivo com timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_resultados = f"resultados/execucao_{timestamp}.txt"

    # Abre arquivo para escrita
    with open(arquivo_resultados, 'w') as arquivo_saida:
        arquivo_saida.write(f"### Experimento de Formação de Grupos ###\n")
        arquivo_saida.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Parâmetros do cenário
        num_alunos = 100
        num_grupos = 5

        arquivo_saida.write(f"Parâmetros do Cenário:\n")
        arquivo_saida.write(f"Número de alunos: {num_alunos}\n")
        arquivo_saida.write(f"Número de grupos: {num_grupos}\n\n")

        # Gera dados sintéticos semelhantes aos do artigo
        # Idade (anos) - uniforme [10, 60]
        # Hora de acesso - uniforme [0, 23]
        # Tempo de acesso (min) - uniforme [1, 90]
        # Disciplina - uniforme [1, 4]
        np.random.seed(42)  # Para reprodutibilidade

        idade = np.random.uniform(10, 60, num_alunos).reshape(-1, 1)
        hora_acesso = np.random.uniform(0, 23, num_alunos).reshape(-1, 1)
        tempo_acesso = np.random.uniform(1, 90, num_alunos).reshape(-1, 1)
        disciplina = np.random.randint(1, 5, num_alunos).reshape(-1, 1)

        # Combina todas as características
        dados_alunos = np.hstack((idade, hora_acesso, tempo_acesso, disciplina))

        # Normaliza os dados
        dados_normalizados = normalizar_dados(dados_alunos)

        print("Iniciando algoritmo genético para formação de grupos...")
        arquivo_saida.write("Iniciando algoritmo genético para formação de grupos...\n")

        # Cria e executa o AG
        ag = AlgoritmoGeneticoAVA(
            dados_normalizados,
            num_grupos,
            tam_populacao=100,
            pc=0.1,
            pm=0.03,
            num_geracoes=100
        )

        melhor_solucao, melhor_fitness = ag.executar(arquivo_saida)
        soma_distancias = 100/melhor_fitness - 1

        print(f"Melhor solução encontrada: {melhor_solucao}")
        print(f"Fitness da melhor solução: {melhor_fitness}")
        print(f"Soma das distâncias: {soma_distancias}")

        arquivo_saida.write("\nResultados do Algoritmo Genético:\n")
        arquivo_saida.write(f"Fitness da melhor solução: {melhor_fitness}\n")
        arquivo_saida.write(f"Soma das distâncias: {soma_distancias}\n\n")

        # Salva a distribuição de alunos por grupo
        arquivo_saida.write("Distribuição de alunos por grupo:\n")
        for g in range(1, num_grupos + 1):
            num_alunos_grupo = np.sum(melhor_solucao == g)
            arquivo_saida.write(f"Grupo {g}: {num_alunos_grupo} alunos\n")

        print("\nIniciando comparação com método aleatório...")
        arquivo_saida.write("\nIniciando comparação com método aleatório...\n")

        # Compara com método aleatório
        media_ag, media_aleatorio = comparar_metodos(dados_alunos, num_grupos, arquivo_saida=arquivo_saida)

        print(f"\nMédia das distâncias (AG): {media_ag:.4f}")
        print(f"Média das distâncias (Aleatório): {media_aleatorio:.4f}")
        print(f"Melhoria: {((media_aleatorio - media_ag) / media_aleatorio) * 100:.2f}%")

        print(f"\nResultados salvos em: {arquivo_resultados}")

        # Plota a evolução do fitness
        ag.plotar_evolucao(arquivo_saida)

