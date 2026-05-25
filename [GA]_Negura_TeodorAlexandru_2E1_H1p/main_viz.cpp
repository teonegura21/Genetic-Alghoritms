#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <fstream>
#include "header_viz.h"

using namespace std;

int main()
{
    vector<vector<bool>> DomeniuDeDefinitie;
    ConvertireDomeniu(DomeniuDeDefinitie);

    // Open output files for trajectory data
    ofstream fi_trajectories("fi_trajectories.csv");
    ofstream bi_trajectories("bi_trajectories.csv");
    ofstream landscape_data("landscape.csv");
    ofstream basins_data("basins.csv");

    // Write headers
    fi_trajectories << "StartPoint,Step,Position,Fitness\n";
    bi_trajectories << "StartPoint,Step,Position,Fitness\n";
    landscape_data << "Position,Fitness\n";
    basins_data << "StartPoint,Algorithm,EndPoint,Steps\n";

    // Generate landscape data
    for (int i = 0; i <= 31; i++)
    {
        landscape_data << i << "," << fitnes(i) << "\n";
    }

    map<int, vector<int>> bazineDeAtractie_BI;
    map<int, vector<int>> bazineDeAtractie_FI;

    cout << "Generating trajectory data for visualization..." << endl;

    for (int i = 0; i <= 31; i++)
    {
        // Best Improvement
        vector<bool> punct_start_BI = DomeniuDeDefinitie[i];
        int pasic_BI = 0;
        vector<int> trajectory_BI;
        HC_BI(punct_start_BI, pasic_BI, trajectory_BI);
        int maxim_gasit_BI = Baza10(punct_start_BI);
        bazineDeAtractie_BI[maxim_gasit_BI].push_back(i);

        // Write BI trajectory
        for (int step = 0; step < trajectory_BI.size(); step++)
        {
            bi_trajectories << i << "," << step << ","
                          << trajectory_BI[step] << ","
                          << fitnes(trajectory_BI[step]) << "\n";
        }
        basins_data << i << ",BI," << maxim_gasit_BI << "," << pasic_BI << "\n";

        // First Improvement
        vector<bool> punct_start_FI = DomeniuDeDefinitie[i];
        int pasic_FI = 0;
        vector<int> trajectory_FI;
        HC_FI(punct_start_FI, pasic_FI, trajectory_FI);
        int maxim_gasit_FI = Baza10(punct_start_FI);
        bazineDeAtractie_FI[maxim_gasit_FI].push_back(i);

        // Write FI trajectory
        for (int step = 0; step < trajectory_FI.size(); step++)
        {
            fi_trajectories << i << "," << step << ","
                          << trajectory_FI[step] << ","
                          << fitnes(trajectory_FI[step]) << "\n";
        }
        basins_data << i << ",FI," << maxim_gasit_FI << "," << pasic_FI << "\n";

        cout << "Processed starting point " << i << endl;
    }

    fi_trajectories.close();
    bi_trajectories.close();
    landscape_data.close();
    basins_data.close();

    cout << "\n=== Data files created successfully! ===" << endl;
    cout << "Files created:" << endl;
    cout << "  - landscape.csv (fitness landscape)" << endl;
    cout << "  - fi_trajectories.csv (First Improvement paths)" << endl;
    cout << "  - bi_trajectories.csv (Best Improvement paths)" << endl;
    cout << "  - basins.csv (basin of attraction data)" << endl;

    cout << "\n--- REZULTAT: Bazine de Atractie (Best Improvement) ---" << endl;
    for (auto const& [maxim, puncte] : bazineDeAtractie_BI)
    {
        cout << "Maximul " << maxim << " (fitness=" << fitnes(maxim)
             << ") [Bazin: " << puncte.size() << " puncte]: ";
        for (int p : puncte) { cout << p << " "; }
        cout << endl;
    }

    cout << "\n--- REZULTAT: Bazine de Atractie (First Improvement) ---" << endl;
    for (auto const& [maxim, puncte] : bazineDeAtractie_FI)
    {
        cout << "Maximul " << maxim << " (fitness=" << fitnes(maxim)
             << ") [Bazin: " << puncte.size() << " puncte]: ";
        for (int p : puncte) { cout << p << " "; }
        cout << endl;
    }

    return 0;
}
