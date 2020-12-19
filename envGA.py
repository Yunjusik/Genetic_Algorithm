import gym
import math
import random
import numpy as np
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
from gaft.analysis import ConsoleOutput
from gaft import GAEngine
from scipy import special


class envGA(gym.Env):
    def __init__(self):
        # CSI parameters
        self.n = 20
        self.m =2 # RAT, LTE:0, 5G midband:1
        self.best_RAT = 0
        self.stack = None
        self.omega = 100
        self.radius = 200
        self.nk_max = 40
        self.U_b_n = 32 * 8
        self.U_c_n = 330 * 32
        self.U_Q_1 = 0.001
        self.U_Q_2 = 10 ** (-7)
        self.Ts = 0.000125

    ''' GA_UE의 인풋은 각UE가 스케쥴링할때 필요한 사전정보들, H, BW를 집어넣음
            U_b_n = 32 * 8
        U_c_n = 330 * 32
        U_Q_1 = 0.001
        U_Q_2 = 10 ** (-7)
    [GA_UE Input]
    H_raw = (40x2)
    BW = 
                    RAT1    RAT2
                  --------- ---------
           urllc | BW_u_R1 | BW_u_R2 |
                  --------- ---------
           eMBB  | BW_e_R1 | BW_e_R2 |       
                  --------- ---------
    [GA_Output]
    Job (40x100x5) --> 요걸 낱개로 5장을 normalize 해서 붙일지 고민. 나중에 normalize 이슈 있음
    GA_UE의 리턴값은 40x100x5 Job matrix,람다u,람다e, 
    '''

    def GA_urllc(self, node_index, lam_u,  H_raw, BW ):

        ''' URLLC 오프로딩 decision
        일단 H를 선택해서, 1 time slot (0.000125s)안에 데이터를 보낼 수 없는경우, 확정 오프로딩을 진행 (o=1, p= 0, f=존재)
        채널게인이 적당히 괜찮은 경우, GA는 EE최소화되는 최적의 f,o,p 조합을 찾음

        구조  if문
        if 채널게인 bad, -> local  처리
        else, 채널게인 good -> 오프로딩 조합 찾기(GA이용)
        즉 이 경우 GA는 채널게인이 나쁜 경우에만 사용.

        '''
        #best_RAT = int(best_RAT[node_index])
        #print(best_RAT)
        #H = H_raw[node_index, best_RAT]
        #print(H_raw.shape) ## 40x2
        #print('Im the BW shape!!:', BW)   ## 2x2
        A = H_raw[node_index,0] * BW[0,0]
        B = H_raw[node_index,1] * BW[0,1]
        if A > B: ## RAT 1이 더 좋을때
            H = H_raw[node_index,0]
            BW =BW[0,0]
        else: ## RAT 2가 더 좋을때
            H = H_raw[node_index,1]
            BW = BW[0,1]
        #print(H)

        #BW = BW[0,best_RAT]

        max_SNR =  (H * 0.2)/(BW * (10**(-20.4)))
        V = 1 - (1 / (1 + max_SNR) ** 2)
        coef = math.sqrt(0.000125 * BW / V) * (np.log(1 + max_SNR) - (32 * 8 * np.log(2)) / (BW * 0.000125))
        error_dec = 0.5 - 0.5 * special.erf(coef / math.sqrt(2))
        SNR_Loss = math.sqrt(V/(0.00015 * BW)) * special.erfcinv(10**-7) * math.sqrt(2)
        max_D_rate = (BW/math.log(2)) * (math.log(1+max_SNR) - SNR_Loss )
        T_delay_min = 32 * 8 / max_D_rate

          ## channel dispersion

        if T_delay_min > 0.000125 or error_dec > 0.5 * (10**-7): ### one time slot 초과시 fully local processing, o=1, p=0

            mutation = FlipBitMutation(pm=0.1)
            selection = TournamentSelection()
            crossover = UniformCrossover(pc=0.8, pe=0.5)
            indv = BinaryIndividual(ranges=[(0, 1)], eps=0.001)
            population = Population(indv_template=indv, size=50).init()
            engine = GAEngine(population=population, selection=selection,
                              crossover=crossover, mutation=mutation,
                              analysis=[ConsoleOutput])
            @engine.fitness_register
            @engine.minimize
            def fitness(indv):

                global done_time
                done_time = False
                global done_error
                done_error = False

                #print('Now',node_index, 'th node do optimization process')

                f, = indv.solution
                pen = 0
                q = lam_u
                D = math.ceil(330*32/125000*(f+0.000001))
                roh = q*D
                ii = math.ceil(8-D)
                jj = math.floor(ii/D)
                SIGMA = 0
                error_u=0
                for k in range(jj + 1):
                    SIGMA = SIGMA + ((q * ((1 - q) ** (D - 1))) ** k) * ((-1) ** k) * math.comb(ii + k - k * D, k)
                    error_u = 1 - ((1 - q) ** (-ii - 1)) * (1 - roh) * SIGMA  # find error

                if D > 8: # 8 time slot = 0.001 second
                    pen = 5 + 100*(D-8)
                    done_time = True

                if error_u > 10**-7:
                    pen = 5 + 100*(error_u - 10**-7)
                    done_error = True

                return (10**(-15))*((125000*f)**2) * (330*32) + pen


            engine.run(ng=100)
            best_indv = engine.population.best_indv(engine.fitness)

            f = best_indv.solution[0]
            Off_bit = 0
            T_d = 0
            done = done_time or done_error
            EE = (10**(-15))*((125000*f)**2) * (330*32)
            data_rate = 0
            error_dec = 0
            '''로컬처리만 하는경우, frequency만찾음'''
            print('URLLC done', done)
            return Off_bit, T_d, EE, data_rate , BW , error_dec,  done

        else:   ## For offloading case,,, done이 나오면 EE 평가하지 않음.
            mutation = FlipBitMutation(pm=0.1)
            selection = TournamentSelection()
            crossover = UniformCrossover(pc=0.8, pe=0.5)
            indv = BinaryIndividual(ranges=[(0, 1), (0, 1), (0, 1)], eps=0.001)
            population = Population(indv_template=indv, size=50).init()
            engine = GAEngine(population=population, selection=selection,
                              crossover=crossover, mutation=mutation,
                              analysis=[ConsoleOutput])
            @engine.fitness_register
            @engine.minimize
            def fitness(indv) : ########self.Ts 써서, 나중에 Ts 바꾸면 여기도 바꿔줘야함...
                global done_time
                done_time = False
                global done_error
                done_error = False
                global done_t_delay
                done_t_delay = False


                f,o,p = indv.solution

                e=10**-7
                f=f+e
                o=o+e
                p=p+e

                pen1=0
                pen2=0
                pen3=0
                pen4=0

                #####################
                SNR =  (0.2 * p * H) / (BW*(10**(-20.4)))
                D_rate = (BW / math.log(2)) * (math.log(1 + SNR))

                V = 1 - (1 / (1 + SNR) ** 2)
                coef = math.sqrt((0.000125 * BW) / (V + e)) * (
                            np.log(1 + SNR) - ((32 * 8 * np.log(2)) / (BW * 0.000125)))
                if coef <= 5.3:
                    coef = 0
                    pen4 = 50

                ################################
                global T_delay

                T_delay = 32 * 8 / D_rate


                q = 0.1*o
                D = math.ceil((330 * 32) / (125000 * (f)))
                roh = q * D
                ii = math.ceil(8 - D)
                jj = math.floor(ii / D)
                SIGMA = 0
                error_u = 0
                for k in range(jj + 1):
                    SIGMA = SIGMA + ((q * ((1 - q) ** (D - 1))) ** k) * ((-1) ** k) * math.comb(ii + k - k * D, k)
                    error_u = 1 - ((1 - q) ** (-ii - 1)) * (1 - roh) * SIGMA  # find error
             #   if coef <= 5.3:
                #    pen4 = 50 + 10 * (5.3 - coef)

                if T_delay > 0.000125: ### time constraints check
                    pen1 = 50 + 10 * (T_delay - 0.000125)
                    done_t_delay = True
                if D > 8:  ## 1ms  QoS condition check
                    pen2 = 50 + 10 * (D - 8)
                    done_time = True

                if error_u > 10 ** -7:  ## 로컬 큐잉 위반 조건 패널티
                    pen3 = 50 + 10 * (error_u - 10 ** -7)
                    done_error = True

                return (10 ** (-15)) * ((125000 * f) ** 2) * o * (330 * 32) + (1 - o) * p * T_delay + pen1+pen2+pen3+pen4

            engine.run(ng=100)
            best_indv = engine.population.best_indv(engine.fitness)
            #print('best_indv.solution',best_indv.solution)
            f = best_indv.solution[0]
            o = best_indv.solution[1]
            p = best_indv.solution[2]
            Off_bit = (1-o)*32*8

            SNR = (0.2 * p * H)/ (BW * 10 ** -20.4)
            V = 1 - (1 / (1 + SNR) ** 2)
            coef = math.sqrt(0.000125 * BW / (V)) * (np.log(1 + SNR) - (32 * 8 * np.log(2)) / (BW * 0.000125))
            error_dec = 0.5 - 0.5 * special.erf(coef / math.sqrt(2))
            #print('Decoding error probability : ', error_dec)

            T_d = T_delay
            done = done_time or done_error or done_t_delay

            EE = (10**(-15))*((125000*(f))**2)*o*(330*32)+ (1-o)*p*T_d
            data_rate = (BW * math.log2(1 + (H * 0.2 * p) / (BW * (10 ** (-20.4)))))
            print('URLLC task done',done)

        return Off_bit, T_d, EE, data_rate , BW , error_dec,  done
        ### Done이 뜨는 경우, EE 는 포함하지말고, SE의 BW 및 dat_rate에는 포함시켜야함.



###################GA_eMBB_v1 요거는 나중에 벤치마킹핡때, GA의 방향성 (goodfit) 설정하고 안하고 벤치마크용으로 해놔야겠음.
    def GA_eMBB(self, node_index, lam_e, H_raw, BW ):
     
        H1  = H_raw[node_index,0]
        H2  = H_raw[node_index,1]
        BW1 = BW[1,0]
        BW2 = BW[1,1]
        ran = random.random()
        b_n = 50000 * (1 + ran) * 8
        Tn_max = 1 * (1 + ran)

        mutation = FlipBitMutation(pm=0.1)
        selection = TournamentSelection()
        crossover = UniformCrossover(pc=0.8, pe=0.5)
        indv = BinaryIndividual(ranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], eps=0.001)
        population = Population(indv_template=indv, size=50).init()
        engine = GAEngine(population=population, selection=selection,
                          crossover=crossover, mutation=mutation,
                          analysis=[ConsoleOutput])
        @engine.fitness_register
        @engine.minimize
        def fitness(indv):
            global done
            done = False
            #global T_tran_1
            #global T_tran_2
            global T_tran
            #global data_rate
            done_local_queue = False
            done_time = False

            o_1, o_2, f, p1, p2 = indv.solution ## o_1,o_2 는 slicing ratio
            pen =0
            pen1 =0
            pen2 =0
            pen3 = 0
            e = 10 ** -7
            o0 = 1 * o_1
            o1 = (1 - o_1) * o_2
            o2 = 1 - o1
            D1 = (BW1 * math.log2(1 + (H1 * 0.1 * (p1)) / (BW1 * (10 ** (-20.4)))))
            D2 = (BW2 * math.log2(1 + (H2 * 0.1 * (p2)) / (BW2 * (10 ** (-20.4)))))


            T_local = (o0*(330 * b_n/8) / (125000 * (f + e))) * 0.000125

            T_tran_1 = (o1 * (b_n )) / (D1 + e)

            T_tran_2 = (o2 * (b_n )) / (D2 + e)

            T_tran = max(T_tran_1, T_tran_2, T_local)

            #T_latency = max(T_local, T_tran)

            if o0 * lam_e * (330 * b_n/8) - 125000 * f > 0: ### workload utilization on local server should be lower than one
                pen = 100 + 100 * (o0 * lam_e * (330 * b_n/8) - 125000 * f)
                done_local_queue = True

            RMS = (abs(T_tran_1-T_tran_2) + abs(T_tran_1-T_local) + abs(T_tran_2-T_local))/3

            if RMS >= 0.001 :
                pen2 = 100+100*RMS

            if T_tran > Tn_max:
                pen1 = 100 + 100*(T_tran - Tn_max)
                done_time = True

            done = done_local_queue or done_time



            return (10**(-15))*((125000*f)**2)*o0*(330*b_n/8) + 0.1*p1*T_tran_1 + 0.1*p2*T_tran_2 + pen + pen1+ pen2

        engine.run(ng=100)
        best_indv=engine.population.best_indv(engine.fitness)
        #global T_tran_1
        #global T_tran_2
        global T_tran
        #global data_rate
        global done
        o_1 = best_indv.solution[0]
        o_2 = best_indv.solution[1]
        f = best_indv.solution[2]
        p1 = best_indv.solution[3]
        p2 = best_indv.solution[4]
        o0 = 1 * o_1
        o1 = (1 - o_1) * o_2
        o2 = 1 - o1
        Off_bit = (1-o0)*b_n
        BW = BW1+BW2
        print('o0,o1,o2,f,p1,p2,done, T_latency', o0,o1,o2,f,p1,p2,done,T_tran)
        D1 = (BW1 * math.log2(1 + (H1 * 0.1 * (p1)) / (BW1 * (10 ** (-20.4)))))
        D2 = (BW2 * math.log2(1 + (H2 * 0.1 * (p2)) / (BW2 * (10 ** (-20.4)))))
        data_rate = D1 + D2
        T_tran_1 = (o1 * (b_n * 8)) / (D1 + 10**-7 )
        T_tran_2 = (o2 * (b_n * 8)) / (D2 + 10**-7)
        EE = (10 ** (-15)) * ((125000 * f) ** 2) * (o0) * (330 * b_n/8) + 0.1 * p1 * T_tran_1 + 0.1 * p2 * T_tran_2

        if done == True: ## 만약 done이면, offloading 안함. task fail
            Off_bit = 0
            T_tran = 0
            EE = 0
            data_rate =0

        return Off_bit, T_tran, EE, data_rate , BW ,Tn_max ,  done

        # 리턴값. 오프로딩된 태스크 크기, time latency,  총 data rate, BW합, 시간제약, done 유무

    def checkjob_u(self, bit, T_delay, Q1, Q2, f_u, roh, dec_error ): ## f_k_u는 서비스레잇값을 직접 받아오자는 취지
        if bit == 0:
            bit = 0.000000000001

        Cnu = 330*bit/8 ## required cpu cycle to process offloaded task
        T_Proc_slot = math.ceil((Cnu)/(f_u)) ## required time slot to process offloaded task
        T_max_slot = Q1/0.000125
        T_tran_slot = math.ceil(T_delay/self.Ts)
        p_error_MEC = roh **(((f_u*(T_max_slot-T_tran_slot+T_Proc_slot))/ Cnu) -1) ## MEC queueing delay violation

        p_error_dec = dec_error
        p_tot_error = 1-(1-p_error_MEC)*(1-p_error_dec)
        done = p_tot_error > Q2  ## error bound
        done = bool(done)
        return done      ## if error> thresold, return True

    def checkjob_e(self, bit, t_tran, Q1, f_e ): ## f_k_u는 서비스레잇값을 직접 받아오자는 취지

        Cne = 330*bit/8 ## required cpu cycle to process offloaded task

        T_tot_delay = t_tran + (Cne/f_e)*self.Ts

        done = T_tot_delay > Q1

        done = bool(done)

        print('T_tot_Delay, t_tran, done', T_tot_delay, t_tran,done)

        return done
