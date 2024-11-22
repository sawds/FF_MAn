import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt


class FormFactor:
    def __init__(self, u, electron_count={'C':6, 'O':8, 'N':7, 'P':15, 'H':1}, solv_gap=3.3, start=0, stop=-1, step=1, bin_width=0.2, calc_ps=True):
        '''
        Code for calculating form factors from molecular dynamics trajectory files.
        This code takes inspiration from SIMtoEXP and the NMRlipids databank tools.
        The advantages of this code are that it can be easily automated (SIMtoEXP is a GUI) and can be run directly on simulation files (NMRlipids requires making a database (I think..))

        :param u:  a MDAnalysis universe of the system of interest
        :param atom_types: a list of all the atom types of interest. they must correspond to the electron count data below. Ideally the atom types would be calculated dynamically, but it is annoying to parse, especially for UA and CG forcefields
        :param solv_gap: The distance from the simulation edge which is assumed to only contain solvent. Measured in Angstroms
        :param start: The frame to start the analysis on
        :param stop: The frame to step the analysis on
        :param step: How many frames to skip. 2 means read in every other frame
        :param bin_width: The binwidth to use when calculating the electron denisty across the z axis (default 0.2 Angstroms).
        :param calc_ps: Boolian, if True we calculate the solvent electron density from the simulation, if False: use the experimental value of 0.333
        '''
        self.u = u
        self.start = start
        self.stop = stop
        self.step = step
        self.z_bin_width = bin_width
        self.solv_gap = solv_gap
        self.atom_types = list(electron_count.keys())
        # add more if you need them, i.e. for united-atom or CG FFs...
        self.electron_count = electron_count
        self.calc_ps = calc_ps

        self.qz_values = np.linspace(0, 1, 1000)  # Fourier q_z range to calculate

        # min_z is the smallest box z axis in the simulation
        self.min_z = min([ts.dimensions[2] for ts in u.trajectory])

        self.n_bins = int(self.min_z / self.z_bin_width)
        self.z_bin_edges = np.linspace(-self.min_z / 2, self.min_z / 2, self.n_bins + 1)
        self.z_bin_centers = (self.z_bin_edges[1:] + self.z_bin_edges[:-1]) / 2


    def run(self):
        '''
        Run tha analysis.
        This is done in several steps:
        i) The electron density across the z axis of the simulation cell is calculated
        ii) The Solvent electron density (ps) is calculated by taking the average electron density of the outer 3 Angstroms of each side of the z axis
        iii) A fourier transform of the total electron density minus the solvent electron density is performed
        iv) The qz value of the first minima is calculated. This can be useful for comparison to experiment.
        :return: None
        '''
        self._calc_e_density()  # calculate total electron density
        if self.calc_ps:
            self._calc_ps()  # calculate bulk solvent electron density
        else:
            self.ps = 0.333  # experimental electron density of water
        self._calc_form_factors()  # fourier transform (total density - bulk solvent density)
        self._get_first_minimum()  # calculate the qz value of the first minima


    def _calc_form_factors(self):

        ff_x = np.linspace(-self.min_z / 2, self.min_z / 2, self.total_e_density.shape[0] + 1)[:-1] + self.min_z / (2 * self.total_e_density.shape[0])  # z bin centers

        k = 0
        bulk = 0
        while k * self.z_bin_width < 0.33:
            bulk += self.total_e_density[k] + self.total_e_density[-k-1]
            k += 1


        bulk /= (2 * k)

        FF_range = np.linspace(0, 1, 1000)
        fa = np.zeros(FF_range.shape[0])
        fb = np.zeros(FF_range.shape[0])


        for j in range(0, self.total_e_density.shape[0]):
            fa += (self.total_e_density[j] - self.ps) * np.cos(FF_range * ff_x[j]) * self.z_bin_width
            fb += (self.total_e_density[j] - self.ps) * np.sin(FF_range * ff_x[j]) * self.z_bin_width

        fourrier_result = np.sqrt(np.multiply(fa, fa) + np.multiply(fb, fb))

        self.form_factors = fourrier_result


    def _calc_e_density(self):
        '''
        Calculate the electron density across the z axis of the simulation
        '''

        # Initialise an array to store the electron densities for each self.z_bin_center
        total_e_density = np.zeros(self.n_bins)
        # Initialise a frame counter - used for averaging over frames
        frames = 0

        # Initialise a dictionary to contain atom counts per bin for each atom type
        self.e_density_profiles = {atom_type: np.zeros(self.n_bins) for atom_type in self.atom_types}

        # The MDAnalysis atom selections for each atom type
        atom_groups = {atom_type: self.u.select_atoms(f'type {atom_type}') for atom_type in self.atom_types}

        for ts in self.u.trajectory[self.start: self.stop: self.step]:
            frames += 1
            box_z = ts.dimensions[2]
            cur_vol = self.z_bin_width * np.prod(ts.dimensions[:2])  # bin volume at the current timestep - accounts for expansion / constriction in the xy plane

            for atom_type, atoms in atom_groups.items():
                z_positions = atoms.positions[:, 2]  # z coords of the current atom type at the current frame
                z_positions_centered = z_positions - (0.5 * box_z)  # center the coords around 0
                atom_counts = np.histogram(z_positions_centered, bins=self.n_bins, range=(-self.min_z/2, self.min_z/2))[0]  # calculate frequency of binned coordinates
                e_counts = atom_counts * self.electron_count[atom_type]  # multiply by electron count for the atom type
                e_density = e_counts / cur_vol  #  Divide by volume for e density
                total_e_density += e_density

        total_e_density /= frames
        self.total_e_density = total_e_density


    def _calc_ps(self):
        '''
        Calculate the bulk solvent density
        This is done by calculating the density of solvent in a slices of the system at the top and bottom of the box
        '''
        # remove the outermost few bins - they can be numerically unstable...
        discarded_bins = 5
        # How many bins are assumed to be full of water, counting outside inwards
        bulk_bins = int(self.solv_gap / self.z_bin_width)
        bulk_e_density = np.concatenate((self.total_e_density[discarded_bins: bulk_bins], self.total_e_density[-bulk_bins-1: -discarded_bins-1]))
        # Calculate the electron density average over solvent bins
        bulk_e_density = np.mean(bulk_e_density)
        self.ps = bulk_e_density


    def _calc_atomistic_xray_scattering_length(self, qz, atom_type):
        '''
        Not currently used, instead this code uses the electron count.
        The method outlined here is what is used by SIMtoEXP, and is explained a bit more in the SIMtoEXP paper.
        The function calculates the scattering length of an atom for a given qz.
        It does this by using a parameterised function, which approximates a fourier transform of the atomistic electron density.
        The parameters contained in the dictionary below are taken from SIMtoEXP.

        :param qz: The qz value for which to calculate the scattering length for.
        :param atom_type: The atom type to calculate the scattering length for (must be in the dictionary below).
        :return f_alpha_qz: The scattering length of a specific atom at a specific qz
        '''

        # parameters from the SIMtoEXP source code
        # Originally from Los Alimos research

        simtoexp_scattering_params = {
            'H': {'a': [0.493, 0.323, 0.14, 0.041], 'b': [10.511, 26.126, 3.142, 57.8], 'c': 0.003},
            'C': {'a': [2.31, 1.02, 1.589, 0.865], 'b': [20.844, 10.208, 0.569, 51.651], 'c': 0.216},
            'O': {'a': [3.0485, 2.2868, 1.5463, 0.867], 'b': [13.2771, 5.7011, 0.3239, 32.9089], 'c': 0.2508},
            'N': {'a': [12.213, 3.132, 2.013, 1.166], 'b': [0.006, 9.893, 28.997, 0.583], 'c': -11.524},
            'P': {'a': [3.049, 2.287, 1.546, 0.867], 'b': [13.277, 5.701, 0.324, 32.909], 'c': 0.251}
        }

        params = simtoexp_scattering_params[atom_type]
        a_values = params['a']
        b_values = params['b']
        c_value = params['c']

        f_alpha_qz = sum(a * np.exp(-b * (qz / (4 * np.pi)) ** 2) for a, b in zip(a_values, b_values)) + c_value
        return f_alpha_qz


    def plot_form_factors(self, showfig=True, savefig=False, figname='form_factors.png'):
        '''
        Plot the form factors
        :param showfig: Show the figure at time of code execution (i.e. plt.show()).
        :param savefig: Save the figure (as form_factors.png)
        :return: None
        '''
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.qz_values, self.form_factors)
        ax.set_xlabel(r'q$_{z}$ / e Å^{-1}')
        ax.set_ylabel(r' |F(q$_{z}$)|')

        if savefig:
            plt.savefig(figname, dpi=300)
        if showfig:
            plt.show()
        plt.close()


    def plot_e_density(self, showfig=True, savefig=False, figname='e_density.png'):
        '''
        Plot the total electron density across the z axis of the simulation.
        :param showfig:  Show the figure at time of code execution (i.e. plt.show()).
        :param savefig:  Save the figure (as e_density.png)
        :return: None
        '''
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(self.z_bin_centers, self.total_e_density, label='Total')
        ax.set_xlabel('Bilayer Normal / Å')
        ax.set_ylabel(r'Average Electron Density / $Å ^{-3}$')
        plt.legend()

        if savefig:
            plt.savefig(figname, dpi=300)
        if showfig:
            plt.show()
        plt.close()


    def save_data(self, filename='form_factor.csv'):
        '''
        Save the form factor data to a csv file, with headers
        :param filename: The name of the file to write to
        :return: None
        '''
        with open(filename, 'w') as f:
            f.write('qz, form factor\n')
            for i in range(len(self.form_factors)):
                f.write(f'{self.qz_values[i]}, {self.form_factors[i]}\n')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--trajectory", help="Path to the trajectory file (i.e. .xtc)")
    parser.add_argument("-s", "--structure", help="Path to the structure file (i.e. .gro)")
    parser.add_argument("--start", default=0, help="frame to start analysis on")
    parser.add_argument("--end", default=-1, help="frame to finish analysis on")
    parser.add_argument("--skip", default=1, help="how many frames to skip")
    parser.add_argument("--no_show_fig", action='store_false', default=True)
    parser.add_argument("--no_save_fig", action='store_false', default=True)
    parser.add_argument("--bin_width", default=0.2, help="The width of bin to use in Angstroms (default: 0.2)")
    parser.add_argument('--no_save_data', action='store_false', default=True)
    args = parser.parse_args()

    #  Reading in the trajectory
    u = mda.Universe(args.structure, args.trajectory)

    # Running the analysis
    form_factors = FormFactor(u, start=args.start, bin_width=args.bin_width)
    form_factors.run()

    # Plotting figures
    form_factors.plot_e_density(showfig=args.no_show_fig, savefig=args.no_save_fig)
    form_factors.plot_form_factors(showfig=args.no_show_fig, savefig=args.no_save_fig)

    # Write the form factors and corresponding qz values to file
    if args.no_save_data:
        form_factors.save_data()

    '''
    Example usage from the command line:
    python calc_FF.py -f traj.xtc -s struct.gro
    
    The code will show and save figures of the electron density and form factors, as well as saving a csv file of the numerical form factor data.
    '''

