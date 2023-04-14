def print_line_info(line, return_dict=False):
	"""
	Prints details of a line
	"""
	import numpy as np
	
	dx = line.x2-line.x1
	dy = line.y2-line.y1
	angle = np.arctan(dy/dx)
	angle_degree = angle *180/np.pi
	
	print('Start:  ({},{}) [px]'.format(line.x1, line.y1))
	print('Finish: ({},{} [px])'.format(line.x2, line.y2))
	print('Width:  {} [px]'.format(line.linewidth))
	print('dx:     {}'.format(dx))
	print('dy:     {}'.format(dy))
	print('Length: {} [px]'.format(line.length))
	print('Angle:  {} [rad]'.format(angle))
	print('     :  {} [degree]'.format(angle_degree))
	if return_dict:
		return {'dx': dx,
		'dy': dy,
		'length': line.length,
		'angle': angle,
		'angle_degree': angle_degree}