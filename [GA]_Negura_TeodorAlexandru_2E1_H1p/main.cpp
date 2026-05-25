#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include "header.h"

using namespace std;

int main()
{
    vector<vector<bool>> DomeniuDeDefinitie;
    ConvertireDomeniu(DomeniuDeDefinitie);

    map<int, vector<int>> bazineDeAtractie_BI;
    map<int, vector<int>> bazineDeAtractie_FI;

    cout << "--- Rularea BI si FI de la fiecare punct (0-31) ---" << endl;

    for (int i = 0; i <= 31; i++)
    {
        vector<bool> punct_start_BI = DomeniuDeDefinitie[i];
        int pasic_BI = 0;
        HC_BI(punct_start_BI, pasic_BI);
        int maxim_gasit_BI = Baza10(punct_start_BI);
        bazineDeAtractie_BI[maxim_gasit_BI].push_back(i);

        vector<bool> punct_start_FI = DomeniuDeDefinitie[i];
        int pasic_FI = 0;
        HC_FI(punct_start_FI, pasic_FI);
        int maxim_gasit_FI = Baza10(punct_start_FI);
        bazineDeAtractie_FI[maxim_gasit_FI].push_back(i);

        // Poti sterge liniile astea de 'cout' daca vrei doar rezultatul final
        cout << "BI: Start " << i << " -> Stop " << maxim_gasit_BI << " (Pasi: " << pasic_BI << ")" << endl;
        cout << "FI: Start " << i << " -> Stop " << maxim_gasit_FI << " (Pasi: " << pasic_FI << ")" << endl;
    }

    cout << "\n--- REZULTAT: Bazine de Atractie (Best Improvement) ---" << endl;
    for (auto const& [maxim, puncte] : bazineDeAtractie_BI)
    {
        cout << "Maximul " << maxim << " [Bazin: " << puncte.size() << " puncte]: ";
        for (int p : puncte) { cout << p << " "; }
        cout << endl;
    }

    cout << "\n--- REZULTAT: Bazine de Atractie (First Improvement) ---" << endl;
    for (auto const& [maxim, puncte] : bazineDeAtractie_FI)
    {
        cout << "Maximul " << maxim << " [Bazin: " << puncte.size() << " puncte]: ";
        for (int p : puncte) { cout << p << " "; }
        cout << endl;
    }

    return 0;
}