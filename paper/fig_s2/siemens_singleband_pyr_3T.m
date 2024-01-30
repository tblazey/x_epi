%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Spectral-Spatial RF Pulse Design for MRI and MRSI MATLAB Package
%
% Authors: Adam B. Kerr and Peder E. Z. Larson
%
% (c)2007-2011 Board of Trustees, Leland Stanford Junior University and
%	The Regents of the University of California.
% All Rights Reserved.
%
% Please see the Copyright_Information and README files included with this
% package.  All works derived from this package must be properly cited.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Modified from demo_general.m

%% Reset SS package globals
ss_globals;


% Pyr/Lac/Bic chemical shifts
df = 0.5e-6;				% Conservative shim requirement
pyr = 172.8e-6;
lac = 184.9e-6;
bic = 162.7e-6;


% Convert to frequency
B0 = 30000;
gamma = 1071;
fspec = B0 * ([(lac - 17.5 * df) (lac + 7 * df) (pyr - df * 0.5) (pyr + df * 0.5) (bic - df * 7) (bic + df * 13.5)] - pyr) * gamma;


% Set up pulse parameters
ang = pi/2;
z_thk = 24;
z_tb = 4;

% Set up spectral/spatial specifications
a = [0 1 0];
d = [0.00375, 0.0002, 0.00375];
ptype = 'ex';
z_ftype='ls';				% Use this to get rid of "Conolly Wings"
z_d1 = 0.01;
z_d2 = 0.01;
f_ctr = [0];
s_ftype = 'min';			% min-phase spectral
ss_type = 'Flyback Half'; 	        % Flyback, symmetric frequency
dbg = 0;				% dbg level
                                        % 0 -none, 1 - little, 2 -lots, ...

% Run opptimization
default_opt = {'Max Duration', 25e-3, ...
	            'Num Lobe Iters', 5, ...
               'Min Order', 1, ...
	            'Spect Correct', 1, ...
	            'SLR', 0, ...
	            'Verse Fraction', 0.1, ...
	            'Nucleus', 'Carbon', ...
               'Max B1', 0.2, ...
               'Max Grad', 6, ...
               'Max Slew', 12, ...
               'B1 Verse', 1};
opt = ss_opt(default_opt);
[g,rf,fs,z,f,mxy] = ss_design(z_thk, z_tb, [z_d1 z_d2], fspec, a*ang, d, ptype, ...
			    z_ftype, s_ftype, ss_type, f_ctr, dbg);

set(gcf,'Name', 'Pyruvate 3T');
ss_save(g,rf,max(ang),z_thk, [], 'Varian', fspec, ang);
fprintf(1,'Hit any key to continue:\n');
disp(max(abs(g)));
disp(max(abs(rf)));
pause;
return

