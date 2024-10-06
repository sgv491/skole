import numpy as np
from types import SimpleNamespace
from itertools import permutations

class Person:
    def __init__(self, id,u, ranking):
        self.id = id
        self.u = u # Utility of working with the partner, ordered by their id
        self.ranking = ranking  # Ordered preferred list of partner ids
        self.partner = None # Chosen partner
        self.k = 0 # Index of the preferred list the person has gotten to if they are the 
        self.k_max = len(ranking) # Maximum index of the preferred list

    def next_offer(self):
        if self.k == self.k_max or self.partner is not None:
            return None        
        else:
            self.k += 1
            return self.ranking[self.k-1]

    def react_to_offer(self,offers,printit = False):
        if self.partner is None:
            u_partner = 0
        else:
            u_partner = self.u[self.partner]

        for i,o in enumerate(offers):
            if o == self.id:
                if self.u[i]>u_partner:
                    self.partner = i
                    u_partner = self.u[i]      

        return self.partner

    def __repr__(self):
        if self.partner is None:
            return f'Person {self.id+1} works alone'
        else:
            return f'Person {self.id+1} works with {self.partner+1}'

class MatchingModel:

    def __init__(self,S=10,M=10):
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        par.S = S
        par.M = M

    def simulate_preferences(self):
        par = self.par
        S = par.S
        M = par.M

        # Simulate preferences
        par.S_pref = np.random.uniform(size=(S,M))
        par.M_pref = np.random.uniform(size=(M,S))
        
        # Create ranking
        par.S_ranking = np.argsort(-par.S_pref,axis=1)
        par.M_ranking = np.argsort(-par.M_pref,axis=1)

        # Create persons
        par.S_list = [Person(i, par.S_pref[i,:], par.S_ranking[i,:]) for i in range(S)]
        par.M_list = [Person(i, par.M_pref[i,:], par.M_ranking[i,:]) for i in range(M)]

    def reset(self):
        par = self.par
        for p in par.S_list+par.M_list:
            p.k = 0
            p.partner = None

    def print_matching(self):
        par = self.par
        print('Matching')

        print('Students:')
        for p in par.S_list:
            print(p)

        print('\nMentors')
        for p in par.M_list:
            print(p)

        matchprint= '{'
        for x,y in self.current_matching():
            if x is not None:
                x += 1
            if y is not None:   
                y += 1
            matchprint += f'({x},{y}), '
        matchprint += '}'
        
        print('{(s:m)} :',matchprint)

    def current_matching(self):
        par = self.par
        
        if par.S < par.M:
            current_matching = [(s.id,s.partner) for s in par.S_list]
        else:
            current_matching = [(m.partner,m.id) for m in par.M_list]

        return current_matching

    def check_matching(self):
        par = self.par

        for s in par.S_list:
            if s.partner is None:
                continue
            else:
                m = par.M_list[s.partner]
                if m.partner != s.id:
                    print(f'Error: {s+1} is not matched with {m+1}')

        for m in par.M_list:
            if m.partner is None:
                continue
            else:
                s = par.S_list[m.partner]
                if s.partner != m.id:
                    print(f'Error: {m+1} is not matched with {s+1}')

        self.check_stability()

    def check_stability(self, print_blocking_pairs=True):
        par = self.par
        for s in par.S_list:
            if s.partner is None:
                partner_u = 0
            else:
                partner_u = s.u[s.partner]

            for i in range(s.k_max):
                if s.u[i]>partner_u: # Does s prefer someone else more than their partner
                    m_other= par.M_list[i]
                    if not (m_other.partner is None):
                        m_other_u = m_other.u[m_other.partner]
                        if m_other.u[s.id] <= m_other.u[m_other.partner]: 
                            continue # No blocking pair

                    if print_blocking_pairs:
                        print(f'Error: student {s.id+1} and mentor {i+1} are a blocking pair')
                    return False
        return True

    def DAA(self, proposers='S', print_matching=True):
        par = self.par
        sol = self.sol 

        recievers = 'M' if proposers == 'S' else 'S'
        P_list = getattr(par,f'{proposers}_list' )
        R_list = getattr(par,f'{recievers}_list')

        # Reset
        self.reset()

        round = 1 
        while True:
            if round==1:
                print('Starting DA algorithm')
            else:
                print(f'Round {round}')

            # Step 1: Proposers send offers
            offers = [p.next_offer() for p in P_list]

            # If all proposers either have an offer or have been rejected by everyone, terminate
            if all(offer is None for offer in offers):
                break

            # Step 2: Receivers react to offers
            for r in R_list:
                r.react_to_offer(offers)

            round += 1
            if round >= 100:
                print('Breaking, more than a 100 rounds')
                break

        assert self.check_stability() ,'Something went wrong DAA matching is not stable'

        if print_matching:
            self.print_matching()

    def all_matchings(self):
        par = self.par
        S = par.S
        M = par.M

        all_matchings = list(permutations(range(M), S))
        return all_matchings

    def find_all_stable_matches(self):
        par = self.par
        sol = self.sol

        S = par.S
        M = par.M

        all_matchings = self.all_matchings()
        sol.stable_matchings = []

        for matching in all_matchings:
            self.reset()
            for s_id, m_id in enumerate(matching):
                par.S_list[s_id].partner = m_id
                par.M_list[m_id].partner = s_id

            if self.check_stability(print_blocking_pairs=False):
                sol.stable_matchings.append(matching)

        return sol.stable_matchings
    
    def calculate_utility(self):
        par = self.par
        sol = self.sol
        
        stable_matchings = sol.stable_matchings
        S_utilities = []
        M_utilities = []

        for matching in stable_matchings:
            S_total = 0
            M_total = 0
            for s_id, m_id in enumerate(matching):
                S_total += par.S_list[s_id].u[m_id]
                M_total += par.M_list[m_id].u[s_id]
            S_utilities.append(S_total / par.S)
            M_utilities.append(M_total / par.M)

        return np.mean(S_utilities), np.mean(M_utilities)
