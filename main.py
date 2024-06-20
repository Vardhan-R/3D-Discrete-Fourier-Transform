from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np, pygame

pygame.init()

width, height = 800, 600 # window width and height

# p = np.random.uniform(-2.5, 2.5, (6, 3)) # 6 random points
# p = [(-2.5, -2.5, -2.5), (1, 0, 2), (2, 2.5, 1.7)] # 3 preset points
p = [(-2, 1, 0), (-1.5, 0, -1), (-1, -1, 0), (-0.5, 0, 1), (0, 1, 0), (0.5, 0, -1), (1, -1, 0), (1.5, 0, 1), (2, 1, 0)]  # 9 preset points

# # helix
# temp_lst_1 = [np.pi * x / 5 for x in range(10)]
# temp_lst_2 = [(np.cos(x), np.sin(x)) for x in temp_lst_1]
# p = np.reshape(np.array([[(temp_lst_2[y][0], temp_lst_2[y][1], x + y / 10) for y in range(10)] for x in range(-5, 5)]), (1, -1, 3))[0]

dt = 4e-3
cyclic_drawing = True
nN = len(p)
all_pts = []
RED, GREEN, WHITE = (1, 0, 0), (0, 1, 0), (1, 1, 1)
running = True
pmb1s = False # previous mouse button 1 state

class Circle:
	"""To draw each epicycle."""

	def __init__(self, colour: tuple, radius: float | int = 1, points: int = 32) -> None:
		self.clr = colour
		self.pts_arr = np.array([(radius * np.cos(2 * np.pi * x / points), radius * np.sin(2 * np.pi * x / points), 0) for x in range(points)])

	def rotate(self, theta: float | int = 0, phi: float | int = 0) -> None:
		r_x = np.array([[1, 0, 0],
						[0, np.cos(theta), -np.sin(theta)],
						[0, np.sin(theta), np.cos(theta)]])
		r_y = np.array([[np.cos(phi), 0, -np.sin(phi)],
						[0, 1, 0],
						[np.sin(phi), 0, np.cos(phi)]])
		r_yx = np.matmul(r_y, r_x)
		self.pts_arr = np.array([np.matmul(r_yx, i) for i in self.pts_arr])

	def show(self, centre: np.ndarray) -> None:
		glColor3fv(self.clr)
		glBegin(GL_LINE_LOOP)
		for i in self.pts_arr:
			glVertex3fv(centre + i)
		glEnd()

def calculatePoint(t: float, freqs: np.ndarray, coeffs: np.ndarray, plane: str = "xy", offset: np.ndarray = np.zeros(3)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Calculates the point at time `t`."""

	circle_centres = []
	pt = 0j

	for i in range(nN):
		temp_pt = coeffs[i] * np.exp(freqs[i] * t * 1j)
		circle_centres.append((pt.real, pt.imag))
		pt += temp_pt
	lines = circle_centres + [(pt.real, pt.imag)]

	circle_centres = np.array(circle_centres)
	lines = np.array(lines)
	zeroes = np.zeros((nN, 1))
	more_zeroes = np.zeros((nN + 1, 1))

	match plane:
		case "xy":
			circle_centres = np.concatenate((circle_centres, zeroes), 1)
			lines = np.concatenate((lines, more_zeroes), 1)
			pt = np.array([pt.real, pt.imag, 0])
		case "xz":
			circle_centres = np.concatenate((circle_centres[:, [0]], zeroes, circle_centres[:, [1]]), 1)
			lines = np.concatenate((lines[:, [0]], more_zeroes, lines[:, [1]]), 1)
			pt = np.array([pt.real, 0, pt.imag])
		case "yz":
			circle_centres = np.concatenate((zeroes, circle_centres), 1)
			lines = np.concatenate((more_zeroes, lines), 1)
			pt = np.array([0, pt.real, pt.imag])

	circle_centres += offset
	lines += offset

	return (pt, circle_centres, lines)

def dispLines(pts: np.ndarray, clr: tuple) -> None:
	glColor3fv(clr) # set the colour

	# draw once
	glBegin(GL_LINES)
	for pt in pts:
		glVertex3fv(pt)
	glEnd()

	# draw again (so that the lines look connected)
	glBegin(GL_LINES)
	for pt in pts[1:]:
		glVertex3fv(pt)
	glEnd()

def dispPoints(pts: list[float | int] | np.ndarray, clr: tuple) -> None:
	glColor3fv(clr) # set the colour

	# draw the points
	glBegin(GL_POINTS)
	for pt in pts:
		glVertex3fv(pt)
	glEnd()

def generateEpicycles(p: list[float | int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Generates the epicycles using the discrete Fourier transform.

	Returns the epicycles in descending order of their radii."""

	freqs = np.arange(np.round(-(nN - 1) / 2), np.round((nN + 1) / 2))
	mat = np.exp(-2j * np.pi * np.outer(freqs, np.arange(nN)) / nN)
	pts = np.array([pt[0] + pt[1] * 1j for pt in p])
	coeffs = mat @ pts / (2 * nN)
	coeffs_mag = np.sqrt(np.array([coeff.real ** 2 + coeff.imag ** 2 for coeff in coeffs]))

	order = np.argsort(coeffs_mag)[::-1]

	return (freqs[order], coeffs[order], coeffs_mag[order])

# take the projection of the points on the x-y, x-z and y-z planes
p_xy = [pt[:2] for pt in p]
p_xz = [(pt[0], pt[2]) for pt in p]
p_yz = [pt[1:] for pt in p]

# generate the epicycles for each of the three planes
freqs_xy, coeffs_xy, coeffs_mag_xy = generateEpicycles(p_xy)
freqs_xz, coeffs_xz, coeffs_mag_xz = generateEpicycles(p_xz)
freqs_yz, coeffs_yz, coeffs_mag_yz = generateEpicycles(p_yz)

t = 0.
scrn = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
gluPerspective(45, width / height, 0.1, 50)
glTranslatef(0, 0, -10)
glPointSize(5)
# theta ==> xz; phi ==> yz

while running:
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	for event in pygame.event.get():
		# quitting
		if event.type == pygame.QUIT:
			running = False

		# moving around using the mouse
		if pygame.mouse.get_pressed()[0]:
			current_mouse_pos = pygame.mouse.get_pos()
			if pmb1s:
				glRotatef(1, current_mouse_pos[1] - prev_mouse_pos[1], current_mouse_pos[0] - prev_mouse_pos[0], 0)
				prev_mouse_pos = current_mouse_pos
			else:
				pmb1s = True
				prev_mouse_pos = current_mouse_pos
		else:
			pmb1s = False

	if t < 2 * np.pi * (1 - int(not cyclic_drawing) / nN): # current drawing
		# calculate the position of the current point, and the epicycle info
		pt_xy, circle_centres_xy, lines_xy = calculatePoint(t, freqs_xy, coeffs_xy, "xy")
		circles_xy = [Circle(GREEN, coeffs_mag_xy[i]) for i in range(nN)]

		pt_xz, circle_centres_xz, lines_xz = calculatePoint(t, freqs_xz, coeffs_xz, "xz", lines_xy[-1])
		circles_xz = [Circle(GREEN, coeffs_mag_xz[i]) for i in range(nN)]

		pt_yz, circle_centres_yz, lines_yz = calculatePoint(t, freqs_yz, coeffs_yz, "yz", lines_xz[-1])
		circles_yz = [Circle(GREEN, coeffs_mag_yz[i]) for i in range(nN)]

		# rotate and display the epicycles
		for i in range(nN):
			circles_xz[i].rotate(np.pi / 2)
			circles_yz[i].rotate(0, np.pi / 2)
			circles_xy[i].show(circle_centres_xy[i])
			circles_xz[i].show(circle_centres_xz[i])
			circles_yz[i].show(circle_centres_yz[i])

		pt = pt_xy + pt_xz + pt_yz
		all_pts.append(pt) # store the current point

		t += dt # update the time

		# display the epicycle lines
		dispLines(lines_xy, GREEN)
		dispLines(lines_xz, GREEN)
		dispLines(lines_yz, GREEN)

		dispPoints(p, RED) # display the original points
		dispPoints([pt], WHITE) # display the current point
	else: # final drawing
		dispLines(all_pts, WHITE) # display all the points, joined with lines
		dispPoints(p, RED) # display the original points

	pygame.display.flip()

pygame.quit()
