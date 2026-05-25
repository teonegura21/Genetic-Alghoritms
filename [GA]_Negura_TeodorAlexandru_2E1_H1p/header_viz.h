#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <fstream>
using namespace std;

int fitnes(int punct)
{
    int fitnes = pow(punct,3) - 60* pow(punct,2) + 900*punct + 100;
    return fitnes;
}

void decodare (int nr, vector<bool> &rez)
{
    //din baza 10 in baza 2
    if (nr%2==0) rez[0]=0;
    else rez[0]=1;

    for (int ii = 4; ii >= 1 ; ii--)
    {
        if( pow(2,ii) <= nr)
        {
            rez[ii]=1;
            nr -= pow(2,ii);
        }
        else continue;
    }
}

void ConvertireDomeniu(vector<vector<bool>> &Definitie)
{
    //face din numerele int de la 0-31 numere pe 5 biti, cu vectori de vectori
    Definitie.resize(32);
    for (int ii=0; ii<=31; ii++)
    {
        vector<bool> pas_curent(5,0);
        decodare(ii , pas_curent);
        Definitie[ii]=pas_curent;
    }
}

int flipbit(bool valoare)
{
    if (valoare == 1)
    {
        return 0;
    }
    else if (valoare == 0 )
    {
        return 1;
    }
}

int Baza10(vector<bool> p)
{
    int valoare= 0;
    for (int ii = 0 ; ii< p.size() ; ii++)
    {
        if(p[ii] == true)
        {
            valoare += pow(2,ii);
        }
    }
    return valoare;
}

bool EsteMaximLocal(vector<bool> pcurent)
{
    int val_init = fitnes(Baza10(pcurent));
    for (int ii = 0 ; ii< pcurent.size() ; ii++)
    {
        pcurent[ii]=flipbit(pcurent[ii]);
        int val_schimbata = fitnes(Baza10(pcurent));
        if(val_init < val_schimbata)
        {
            return 0;
            break;
        }
        pcurent[ii]=flipbit(pcurent[ii]);
    }
    return 1;
}

// Modified HC_BI to track trajectory
void HC_BI(vector<bool>& pcurent, int& puncte_vizitate, vector<int>& trajectory)
{
    trajectory.push_back(Baza10(pcurent));
    puncte_vizitate++;

    int fitness_curent = fitnes(Baza10(pcurent));

    int best_neighbor_fitness = -1;
    int best_neighbor_index = -1;

    for (int i = 0; i < 5; i++)
    {
        pcurent[i] = flipbit(pcurent[i]);
        int fitness_vecin = fitnes(Baza10(pcurent));

        if (fitness_vecin > best_neighbor_fitness)
        {
            best_neighbor_fitness = fitness_vecin;
            best_neighbor_index = i;
        }
        pcurent[i] = flipbit(pcurent[i]);
    }

    if (best_neighbor_fitness <= fitness_curent)
    {
        return;
    }

    pcurent[best_neighbor_index] = flipbit(pcurent[best_neighbor_index]);

    HC_BI(pcurent, puncte_vizitate, trajectory);
}

// Modified HC_FI to track trajectory
void HC_FI(vector<bool>& pcurent, int& puncte_vizitate, vector<int>& trajectory)
{
    trajectory.push_back(Baza10(pcurent));
    puncte_vizitate++;

    int fitness_curent = fitnes(Baza10(pcurent));

    for (int i = 0; i < 5; i++)
    {
        pcurent[i] = flipbit(pcurent[i]);
        int fitness_vecin = fitnes(Baza10(pcurent));

        if (fitness_vecin > fitness_curent)
        {
            HC_FI(pcurent, puncte_vizitate, trajectory);
            return;
        }

        pcurent[i] = flipbit(pcurent[i]);
    }

    return;
}
