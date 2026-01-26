import numpy as np
import math
import random
import time
from scipy.spatial.distance import pdist, squareform

class RealCodecGA_JGG_SPX:
    '''
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）

    初期化
    交叉する個体の選択
    交叉
    残す個体の選択
    '''
    def __init__(self, gene_num, 
                 initial_min, initial_max, 
                 population, 
                 crossover_num, child_num,
                 evaluation_func, 
                 seed=None):
        #乱数シード
        np.random.seed(seed=seed)

        # 遺伝子情報
        self.gene_num = gene_num

        #初期化
        self.initial_min = initial_min
        self.initial_max = initial_max

        # 交叉する個体の選択
        self.population = population
        self.crossed_num = crossover_num

        # 交叉
        self.child_num = child_num

        # 残す個体の選択
        # evaluation function defined as following.
        #  func(genes):
        #    return evaluations
        self.evaluation_func = evaluation_func

        #
        self.genes = None # 候補の遺伝子
        self.evals = None # 候補の遺伝子の評価値
        self.best_gene = None # 現時点で評価値が一番高い遺伝子
        self.best_evaluation = None # 現時点で一番高い評価値

        #
        self.__initialize_genes()

        return

    def __initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)
        
        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx]

        return

    def generation_step(self):
        # 個体数
        pop = len(self.genes)

        # 交叉する個体の選択
        # 交叉する個体のインデックス
        crossed_idx = self.__random_select(pop, self.crossed_num)

        # 交叉(spx)
        child_genes = self.__simplex_crossover(self.genes[crossed_idx], self.child_num)
        
        #交叉(undx-m)エラーでる
        #child_genes = self.__unimodal_normal_distribution_crossover_m(self.genes[crossed_idx], self.child_num)
        

        #交叉(rex)
        #child_genes = self.__rexstar_crossover(self.genes[crossed_idx], self.child_num)
        
        
        #残す個体の選択
        survive_genes, survive_evals = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
        #
        self.genes[crossed_idx] = survive_genes
        self.evals[crossed_idx] = survive_evals

        # 最高評価の更新
        if self.best_evaluation > np.min(survive_evals):
            min_idx = np.argmin(survive_evals)
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx]

        return

    def calc_diversity(self):
        dvs = np.average(np.std(self.genes, axis=0))
        return dvs

    @staticmethod
    def __random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def __simplex_crossover(genes, child_num):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # expansion_rate
        expansion_rate = np.sqrt(crss_num + 1.0)

        # Qk計算, Q_(k-1) = e * (P_(k-1) - P_(k)), k=1,2,...,n
        # n = crss_num-1
        diff_genes = np.diff(genes, n=1, axis=0) * (-1.0) * expansion_rate

        # R行列計算
        # [[uniform]*(crsvr_num-1)]*child_num
        rand_coef = np.random.rand(child_num, crss_num-1)
        rand_coef = np.power(rand_coef, 1.0/(np.arange(crss_num-1)+1.0))
        # [u_0*u_1*...*u_n, u_1*...*u_n, ... , u_(n-1)*u_n, u_n
        rand_coef = np.cumprod(rand_coef[:,::-1], axis=1)[:,::-1]

        # Cn計算
        Cn = np.dot(rand_coef, diff_genes)

        # xn計算
        center_gravity = np.average(genes, axis=0)
        xn = center_gravity + expansion_rate * (genes[-1] - center_gravity)

        # C計算
        child_genes = xn + Cn

        return child_genes
    
    def __unimodal_normal_distribution_crossover_m(genes, child_num):
        dim = genes.shape[1]
        N_parents = genes.shape[0]
        
        midpoint_vector = genes.sum(axis=0)/N_parents
        
        difference_vector = parents-midpoint_vector
        
        xi = np.random.normal(loc=0, 
                              scale=1/dim**0.5, 
                              size=(child_num,dim))
        children = midpoint_vector + np.dot(xi,difference_vector[:-1])
        
        return child_genes

    @staticmethod
    def __ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        #"""
        survive_genes = genes[survive_idx]
        survive_eval = evals[survive_idx]
        #"""
        sorted_idx = np.argsort(evals[survive_idx])
        survive_genes = genes[survive_idx][sorted_idx]
        survive_eval = evals[survive_idx][sorted_idx]
        return survive_genes, survive_eval
    
    @staticmethod
    def real_coded_ensemble_crossover_star(parents, N_children,evals):
        '''
        REX-star
        parentsをdim+1個体選択して子個体を生成。
        t = 6で仮設定
        '''
        parents = np.array(parents)
        N_parents = parents.shape[0]
        dim = parents.shape[1]
        
        midpoint_vector = parents.sum(axis=0)/N_parents
        difference_vector = parents-midpoint_vector
        parents_mirror = parents - 2*difference_vector
        
        parents_w_mirror = np.vstack((parents, parents_mirror))
        fitnesses = (self.evals(parents_w_mirror)*self.weights).flatten()
        
        parents_high_fitness = parents_w_mirror[fitnesses.argsort()][N_parents:]
        midpoint_vector_high_fitness = parents_high_fitness.sum(axis=0)/N_parents
        direction = midpoint_vector_high_fitness - midpoint_vector
        
        t = 6
        xi_t = np.random.rand(N_children,dim)*t
        xi = (np.random.rand(N_children,N_parents)*2-1)*(3/(N_parents))**0.5
        children = midpoint_vector + xi_t*direction \
                   + np.dot(xi,difference_vector)
        return children

class RealCodecGA_JGG_AREX:
    '''
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）

    初期化
    交叉する個体の選択
    交叉
    残す個体の選択
    '''
    def __init__(self, gene_num, 
                 evaluation_func, 
                 initial_min, initial_max, 
                 population=None, 
                 crossover_num=None, child_num=None, 
                 initial_expantion_rate=None, learning_rate=None, 
                 seed=None):
        '''
        recommended value of parameter are following.
         crossover_num = gene_num + 1
         child_num = 4 * gene_num (~4 * crossover_num)
         initial_expantion_rate = 1.0
         learning_rate = 1/5/gene_num
        '''

        #乱数シード
        np.random.seed(seed=seed)

        # 遺伝子情報
        self.gene_num = gene_num

        # 個体の評価関数
        # evaluation function defined as following.
        #  func(genes):
        #    return evaluations
        self.evaluation_func = evaluation_func

        #遺伝子の初期値範囲
        self.initial_min = initial_min
        self.initial_max = initial_max

        # 全個体数
        # 交叉する個体数
        if (population is not None) and (crossover_num is not None):
            self.population = population
            self.crossed_num = crossover_num
        elif (population is None) and (crossover_num is not None):
            self.crossed_num = crossover_num
            self.population = self.crossed_num
        elif (population is not None) and (crossover_num is None):
            self.population = population
            self.crossed_num = np.minimum(gene_num+1, population)
        elif (population is None) and (crossover_num is None):
            self.crossed_num = gene_num+1
            self.population = self.crossed_num

        # 交叉して生成する子個体数
        self.child_num = child_num if child_num is not None else 4*self.crossed_num

        # 拡張率
        self.expantion_rate = initial_expantion_rate if initial_expantion_rate is not None else 1.0
        # 拡張率の学習率
        self.learning_rate = learning_rate if learning_rate is not None else 1.0/5.0 / gene_num
        
        #開始時間
        t0 = time.time() 

        #
        self.genes = None # 候補の遺伝子
        self.evals = None # 候補の遺伝子の評価値
        self.best_gene = None # 現時点で評価値が一番高い遺伝子
        self.best_evaluation = None # 現時点で一番高い評価値
        self.last_gene = None # 最終時点での遺伝子
        self.child_genes = None
        self.time = t0
        self.runtime = 0
        self.child = 0
        self.es = None
        self.histime = [0]
        self.histime = np.array(self.histime)
        self.hisevals = [0]
        self.hisevals = np.array(self.hisevals)
        self.wG = [0]
        self.hisevals = np.array(self.wG)
        self.evaN = 0
        self.zone0 = 0
        self.zone1 = 0
        self.count_greater = 0
        self.survive = None
        self.mae = 0
        self.ato = 0
        self.dekamae = None
        self.deka = None
        self.uwo = None
        self.i = 0
        
        self._cs = []
        self.senyu = None
        
        #
        self.__initialize_genes()

        return

    def __initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        #通常の初期値生成
        #"""
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)
        #"""
        #カオスの初期値生成
        """
        N = self.population*self.gene_num
        C=np.zeros(N)
        CS=np.zeros(N)
        C[0]=random.uniform(0, 1)
        for i in range(1,N):
            if C[i-1] == 0 or C[i-1] == 0.25 or C[i-1] == 0.5 or C[i-1] == 0.75 or C[i-1] == 1:
                C[i] = 0.85*C[i-1]+0.1*random.uniform(0, 1)
            elif 0 <= C[i-1] and C[i-1] <=0.5:
                C[i] = 2*C[i-1]
            elif 0.5 < C[i-1] and C[i-1] <=1:
                C[i] = 2*(1-C[i-1])
        sorted_C = np.sort(C)
        maxc = sorted_C[N-1]
        minc = sorted_C[0]
        for i in range(0,N):
            CS[i] = self.initial_min+(self.initial_max-self.initial_min)*(C[i]-minc)/maxc-minc
        
        CS = CS.reshape(self.population, self.gene_num)
        self.genes = CS
        """
            
        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx].copy()

        return self.genes

    def generation_step(self,i):
        #t0 = time.time()
        # 個体数
        pop = len(self.genes)
        
        #SGS選択
        """
        g1,g2,g3,elite = self.sgs(self.genes,pop,self.evaluation_func,20)
        
        cx1 = self.HNDDXBOI(g1,g3,self.runtime,10,20)
        
        cx2 = self.SCX(g2,g3,20)
        """

        # 交叉する個体の選択
        #交叉する個体のインデックス
        #"""
        crossed_idx = self.__random_select(pop, self.crossed_num)

        #交叉(AREX)
        #"""
        child_genes, rand_mtrx,wG2 = self.__arex_crossover(self.genes[crossed_idx], self.evals[crossed_idx], self.child_num, self.expantion_rate)
        np.set_printoptions(suppress=True)
        self.wG = wG2
        #"""
        
        #交叉(rexstar)
        """
        rexeval = self.evaluation_func(self.__rexstar_mirror(self.genes[crossed_idx], self.child_num))
        child_genes = self.__rexstar_crossover(self.genes[crossed_idx], rexeval, self.child_num)
        """
        
        #交叉(rexstar+CS)
        """
        rexeval = self.evaluation_func(self.__rexstar_mirror(self.genes[crossed_idx], self.child_num))
        child_genes,wG2,rand_mtrx = self.__rexstar_crossover_CS(self.genes[crossed_idx], rexeval, self.child_num,self.evals[crossed_idx])
        self.wG = wG2
        """
        
        # 交叉(spx)
        """
        child_genes = self.__simplex_crossover(self.genes[crossed_idx], self.child_num)
        """
        
        t1=time.time()
        self.runtime = t1 - self.time
        #self.runtime = t1 - t0 + self.runtime
        #self.histime = np.append(self.histime,self.runtime)
        
        #AREX用ES
        """
        child_genes = self.ES(child_genes)
        """
        
        
        #CM演算子
        """
        child_genes = np.append(cx1,cx2,axis=0)
        if i%3==1:
            xm = self.Levy(child_genes,self.gene_num)
        elif i%3==2:
            xm = self.CaM(child_genes,self.gene_num)
        elif i%3==0:
            xm = self.DM(child_genes,self.runtime,10,g1,self.gene_num)
        """
        
        #AREX用CM演算子
        """     
        if i%3==1:
            xm = self.Levy(child_genes,self.gene_num)
        elif i%3==2:
            xm = self.CaM(child_genes,self.runtime,0.1,self.gene_num)
        elif i%3==0:
            xm = self.DM(child_genes,self.runtime,0.1,child_genes[0],self.gene_num)
        """
        
        
        #カオス
        #cs = self.tent(child_genes, 100, self.runtime, 5,-5,self.child_num)
        
        #子個体とカオスの結合
        #child_genes = np.append(child_genes,cs,axis=0)
        
        
        #elite保存
        """
        child_genes = np.append(child_genes,xm,axis=0)
        survive_genes, survive_evals, survive_idx = self.Elite_Pre(elite,child_genes,pop,self.evaluation_func)
        """
        
        # 残す個体の選択
        #"""
        survive_genes, survive_evals, survive_idx = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
        self.zone0 = survive_idx
        #"""
        
        #拡張率の更新(arex)
        #"""
        self.expantion_rate = self.__update_arex_expantion_rate(self.expantion_rate, self.learning_rate, rand_mtrx[survive_idx], crss_num=self.crossed_num)
        #print(self.expantion_rate)
        #"""
        
        
        #"""
        if i == 0 or self.count_greater > 0:
        #if i < 1000:
            #self.mae = len(child_genes)
            #child_genes_cm = np.append(child_genes,xm,axis=0)
            cs = self.tent(child_genes, 0.1, self.runtime, self.initial_max,self.initial_min,self.child_num,self.gene_num,self.wG)
            #self.mae = len(child_genes)
            child_genes = np.append(child_genes,cs,axis=0)
            #self.ato = len(child_genes)
            survive_genes, survive_evals, survive_idx = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
            
            
        # カウント用変数
        self.count_greater = 0  # child_numより大きい
        count_less = 0     # child_numより小さい

        # 判定ループ
        for idx in survive_idx:
            if idx > self.child_num:
                self.count_greater += 1
            elif idx <= self.child_num:
                count_less += 1
        self.zone1 = survive_idx
        self.survive = survive_genes
        #"""
        
        # 親個体と変更
        #"""
        self.genes[crossed_idx] = survive_genes.copy()
        self.evals[crossed_idx] = survive_evals
        #"""       

        min_idx = np.argmin(survive_evals)
        # 最終遺伝子の更新
        self.last_gene = survive_genes[min_idx].copy()
        # 生き残ったうちの最高評価
        best_survive_evals = survive_evals[min_idx]
        self.hisevals = np.append(self.hisevals,best_survive_evals)
        # 最高評価の更新
        if self.best_evaluation > survive_evals[min_idx]:
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx].copy()
            
        #self.var = self.population_variance(self.genes)
        #self.var = self.population_variance(survive_genes)
        
        #self.senyu = self.count_clusters(self.genes, eps=0.1, min_samples=3)
        
        #self.senyu = self.count_clusters(survive_genes, eps=1, min_samples=3)
        
        return best_survive_evals

    def calc_diversity(self):
        dvs = np.average(np.std(self.genes, axis=0))
        return dvs

    @staticmethod
    def __random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def __arex_crossover(genes, evals, child_num, expantion_rate):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # 評価が高い順に並べたインデックス
        sorted_idx = np.argsort(evals)

        # 荷重重心
        w = 2.0 * (crss_num + 1.0 - np.arange(1, crss_num+1)) / (crss_num * (crss_num + 1.0))
        wG = np.dot(w[np.newaxis,:], genes[sorted_idx])

        # 重心
        G = np.average(genes, axis=0)

        # 乱数
        rnd_mtrx = np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num))
        
        # 子個体
        child_genes = wG + expantion_rate * np.dot(rnd_mtrx, genes - G)
        
        #カオス用
        #rnd_mtrx = np.append(rnd_mtrx,np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num)),axis=0)
        
        #CM用
        #rnd_mtrx = np.append(rnd_mtrx,np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num)),axis=0)

        return child_genes, rnd_mtrx,wG
    
    @staticmethod
    def __rexstar_crossover(genes, evals, child_num):
        parents = np.array(genes)
        N_parents = parents.shape[0]
        dim = parents.shape[1]
        N_children = child_num
        
        midpoint_vector = parents.sum(axis=0)/N_parents
        difference_vector = parents-midpoint_vector
        parents_mirror = parents - 2*difference_vector
        
        parents_w_mirror = np.vstack((parents, parents_mirror))
        fitnesses  = evals.flatten()*-1
        
        parents_high_fitness = parents_w_mirror[fitnesses.argsort()][N_parents:]
        midpoint_vector_high_fitness = parents_high_fitness.sum(axis=0)/N_parents
        direction = midpoint_vector_high_fitness - midpoint_vector
        
        t = 6
        xi_t = np.random.rand(N_children,dim)*t
        xi = (np.random.rand(N_children,N_parents)*2-1)*(3/(N_parents))**0.5
        children = midpoint_vector + xi_t*direction \
                   + np.dot(xi,difference_vector)
        
        return children
    
    @staticmethod
    def __rexstar_crossover_CS(genes, evals, child_num, arexeva):
        parents = np.array(genes)
        N_parents = parents.shape[0]
        dim = parents.shape[1]
        N_children = child_num
        
        midpoint_vector = parents.sum(axis=0)/N_parents
        difference_vector = parents-midpoint_vector
        parents_mirror = parents - 2*difference_vector
        
        parents_w_mirror = np.vstack((parents, parents_mirror))
        fitnesses  = evals.flatten()*-1
        
        parents_high_fitness = parents_w_mirror[fitnesses.argsort()][N_parents:]
        midpoint_vector_high_fitness = parents_high_fitness.sum(axis=0)/N_parents
        direction = midpoint_vector_high_fitness - midpoint_vector
        
        t = 6
        xi_t = np.random.rand(N_children,dim)*t
        xi = (np.random.rand(N_children,N_parents)*2-1)*(3/(N_parents))**0.5
        children = midpoint_vector + xi_t*direction \
                   + np.dot(xi,difference_vector)
        
        
        # 交叉される遺伝子数
        crss_num = len(genes)

        # 評価が高い順に並べたインデックス
        sorted_idx = np.argsort(arexeva)
        
        # 乱数
        rnd_mtrx = np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num))
        
        w = 2.0 * (crss_num + 1.0 - np.arange(1, crss_num+1)) / (crss_num * (crss_num + 1.0))
        wG = np.dot(w[np.newaxis,:], genes[sorted_idx])
        
        return children, wG, rnd_mtrx
    
    @staticmethod
    def __rexstar_mirror(genes, child_num):
        parents = np.array(genes)
        dim = len(genes)
        N_parents  = genes.shape[0]
        
        midpoint_vector = parents.sum(axis=0)/N_parents
        difference_vector = parents-midpoint_vector
        parents_mirror = parents - 2*difference_vector
        
        parents_w_mirror = np.vstack((parents, parents_mirror))
        
        return parents_w_mirror

    @staticmethod
    def __update_arex_expantion_rate(expantion_rate, learning_rate, survive_rand_mtrx, crss_num=None):
        survive_num = len(survive_rand_mtrx)
        #
        crss_num_ = crss_num if crss_num is not None else survive_num
        #
        ave_r = np.average(survive_rand_mtrx, axis=0)
        L_cdp = expantion_rate**2 * (survive_num - 1) * (np.sum(np.square(ave_r)) - (np.sum(ave_r))**2 / survive_num)
        L_ave = expantion_rate**2 * (1/(crss_num_ - 1)) * (survive_num - 1)**2 / survive_num
        #
        new_expantion_rate = np.maximum(1.0, expantion_rate * np.sqrt((1.0 - learning_rate) + learning_rate * L_cdp / L_ave))

        return new_expantion_rate

    @staticmethod
    def __ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        #
        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]
        return survive_genes, survive_eval, survive_idx
    
    @staticmethod
    def __ranking_survival_with_sharing(genes, evaluation_func, survive_num, sigma_share=0.5, alpha=1.0):
        genes = np.asarray(genes)
        evals = np.asarray(evaluation_func(genes))  # shape = (N,)
        N = len(genes)

        # 1. 個体間距離行列を作成
        # dist[i][j] = ||genes[i] - genes[j]||_2
        diff = genes[:, np.newaxis, :] - genes[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # shape = (N, N)

        # 2. シェアリング係数行列の計算
        sharing_matrix = np.where(
            dists < sigma_share,
            1.0 - (dists / sigma_share) ** alpha,0.0)

        # 3. シェアされた適応度の計算
        sharing_sum = np.sum(sharing_matrix, axis=1)
        sharing_sum[sharing_sum == 0] = 1.0  # 割り算回避

        shared_fitness = evals / sharing_sum  # 小さい方が良い（最小化）

        # 4. shared fitness に基づきランキング
        survive_idx = np.argpartition(shared_fitness, survive_num)[:survive_num]
        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]

        return survive_genes, survive_eval, survive_idx
    
    
    def HNDDXBOI(self,g1,g3,runtime,maxtime,gene_num):
        i = len(g3)
        D = gene_num
        R = np.random.rand(D)
        at = 2.5 - 2*runtime/maxtime
        #x2 = g1+np.dot(np.dot(at,R),(g1-g3))
        atR = np.dot(at,R)

        x2 = np.zeros((i, D))
        x1 = np.zeros((i, D))
        for i in range(i):
            x2[i] = g1+np.dot(atR,(g1-g3[i]))
            x1[i] = np.random.normal(g1, 0.01+((g1-g3[i])/15)**2, size=(6,))
        x = np.concatenate((x1, x2), axis=0)
        return x
    
    def SCX(self,g2,g3,gene_num):
        D = gene_num
        i = len(g2)
        r1 = np.random.uniform(-np.pi/2, np.pi/2)
        r2 = np.random.uniform(0, np.pi)
        x1 = np.zeros((i, D))
        x2 = np.zeros((i, D))
        for i in range(i):
            x1[i] = (1-math.sin(r1))*g2[i]+math.sin(r1)*g3[i]
            x2[i] = (1-math.cos(r2))*g2[i]+math.cos(r2)*g3[i]
        x = np.concatenate((x1, x2), axis=0)
        return x
    
    def tent(self,child_genes, maxtime, runtime, Ub, Lb,child_num,gene_num,wG):
        D = gene_num
        N = child_num*D
        C=np.zeros(N)
        CS=np.zeros(N)
        Xc=np.zeros(N)
        C[0]=random.uniform(0, 1)
        for i in range(1,N):
            if C[i-1] == 0 or C[i-1] == 0.25 or C[i-1] == 0.5 or C[i-1] == 0.75 or C[i-1] == 1:
                C[i] = 0.85*C[i-1]+0.1*random.uniform(0, 1)
            elif 0 <= C[i-1] and C[i-1] <=0.5:
                C[i] = 2*C[i-1]
            elif 0.5 < C[i-1] and C[i-1] <=1:
                C[i] = 2*(1-C[i-1])
        sorted_C = np.sort(C)
        maxc = sorted_C[N-1]
        minc = sorted_C[0]
        """
        for i in range(0,N):
            CS[i] = Lb+(Ub-Lb)*(C[i]-minc)/maxc-minc
        """
        #"""
        for i in range(0, N):
            if i == 0:
                Ub -= wG[0][0]
                Lb -= wG[0][0]
                CS[i] = Lb + (Ub - Lb) * (C[i] - minc) / (maxc - minc)
                a = 1
            elif i == (child_num*2)*a:
                Ub -= wG[0][a]
                Lb -= wG[0][a]
                CS[i] = Lb + (Ub - Lb) * (C[i] - minc) / (maxc - minc)
                a = a+1
            else :
                CS[i] = Lb + (Ub - Lb) * (C[i] - minc) / (maxc - minc)
        #"""
        CS = CS.reshape(child_num, D)
        G = np.average(child_genes, axis=0)
        ut = 1 - runtime/maxtime
        utt = 1 - ut
        Xc = utt*(G-child_genes)+ut*CS
        
        return Xc
    
    def Elite_Pre(self,elite,child_genes,pop,evaluation_func):
        genes = np.concatenate((elite,child_genes),axis=0)
        
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, pop)[:pop]

        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]
        
        return survive_genes, survive_eval, survive_idx 
    
    def Levy(self,genes,gene_num):
        i = len(genes)
        D = gene_num
        beta = 1
        sigma = (math.gamma(1 + beta) / beta * math.gamma((1 + beta) / 2)) * (math.sin((beta * math.pi) / 2) / 2 ** math.gamma((1 + beta) / 2))
        mu = np.random.normal(0, sigma, i)
        nu = np.random.randn(i)
        s =  mu / abs(nu ** math.gamma(1 + beta))
        ramda = 1
        L = ((ramda * math.gamma(beta) * math.sin((math.pi) * ramda / 2)) / math.pi) * (1 / s ** (1 + beta))
        a = 0.01
        dot = np.dot(a,L)
        xm = np.zeros((i,D))
        for i in range(i):
            xm[i] = genes[i]+dot[i]
        
        return xm

    def CaM(self,genes,runtime,maxtime,gene_num):
        i = len(genes)
        D = gene_num
        F1 = 0.9-(0.9-0.4)*(runtime/maxtime)
        c = math.tan((math.pi)*(F1-(1/2)))
        F = (1/math.pi)*math.atan(c)+1/2
        xm = np.zeros((i,D))
        for i in range(i):
            xm[i] = genes[i]+genes[i]*c
        
        return xm
    
    def DM(self,genes,runtime,maxtime,g1,gene_num):
        i = len(genes)
        D = gene_num
        xm = np.zeros((i,D))
        V = np.random.rand(D)
        F = 0.9-(0.9-0.4)*(runtime/maxtime)
        for j in range(i):
            random_indices = random.sample([index for index in range(i) if index != j], 2)
            selected_genes = [genes[random_indices[0]], genes[random_indices[1]]]
            xm[j] = genes[j] + np.dot(V, (g1 - genes[j]))+F*(selected_genes[0]-selected_genes[1])
        
        return xm
    
    def ES(self,genes):
        error_tolerance = 0.0000001
        num_cols = len(genes[0])
        for col in range(num_cols):
            for i in range(len(genes)):
                for j in range(i + 1, len(genes)):
                    if abs(genes[i][col] - genes[j][col]) <= error_tolerance:
                        genes[j][col] = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand()
        return genes
        
        
    @staticmethod
    def __simplex_crossover(genes, child_num):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # expansion_rate
        expansion_rate = np.sqrt(crss_num + 1.0)

        # Qk計算, Q_(k-1) = e * (P_(k-1) - P_(k)), k=1,2,...,n
        # n = crss_num-1
        diff_genes = np.diff(genes, n=1, axis=0) * (-1.0) * expansion_rate

        # R行列計算
        # [[uniform]*(crsvr_num-1)]*child_num
        rand_coef = np.random.rand(child_num, crss_num-1)
        rand_coef = np.power(rand_coef, 1.0/(np.arange(crss_num-1)+1.0))
        # [u_0*u_1*...*u_n, u_1*...*u_n, ... , u_(n-1)*u_n, u_n
        rand_coef = np.cumprod(rand_coef[:,::-1], axis=1)[:,::-1]

        # Cn計算
        Cn = np.dot(rand_coef, diff_genes)

        # xn計算
        center_gravity = np.average(genes, axis=0)
        xn = center_gravity + expansion_rate * (genes[-1] - center_gravity)

        # C計算
        child_genes = xn + Cn
        
        return child_genes
    
    @staticmethod
    def population_variance(population):
        return np.var(population, axis=0)
    
    @staticmethod
    def count_clusters(population, eps=0.5, min_samples=3):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(population)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return n_clusters, labels


class Sample:
    @staticmethod
    def sample_RealCodecGA_JGG_SPX():
        # define evaluation func
        def minus_l2(ndar):
            ev = - np.sum(np.square(ndar), axis=1)
            return ev
        # rcga class
        rcga = RealCodecGA_JGG_SPX(gene_num=2, 
                                   initial_min=-1, initial_max=1, 
                                   population=10, 
                                   crossover_num=3, child_num=10,
                                   evaluation_func=minus_l2, 
                                   seed=1000)

        # optimize
        for i in range(100):
            rcga.generation_step()
            print('{0}'.format(i+1))
            print(' best evals : {0}'.format(rcga.best_evaluation))
            print(' best genes : {0}'.format(rcga.best_gene))
        return
    @staticmethod
    def sample_RealCodecGA_JGG_AREX():
        # define evaluation func
        def minus_l2(ndar):
            ev = - np.sum(np.square(ndar), axis=1)
            return ev

        # rcga class
        rcga = RealCodecGA_JGG_AREX(gene_num=2, 
                                    evaluation_func=minus_l2, 
                                    initial_min=-1, initial_max=1, 
                                    population=10, 
                                    crossover_num=3, child_num=6, 
                                    initial_expantion_rate=1.0, learning_rate=0.05, 
                                    seed=1000)
        # optimize
        for i in range(100):
            rcga.generation_step()
            print('{0}'.format(i+1))
            print(' best evals : {0:.10f}'.format(rcga.best_evaluation))
            print(' best genes : {0:.10f}'.format(rcga.best_gene))
        return

if __name__ == '__main__':
    #test()
    import verification
    
    #verification.test_Minimize_L2_1()
    verification.test_LinearLeastSquares_withRCGA()
    
