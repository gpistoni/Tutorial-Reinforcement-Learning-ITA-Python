import pygame
import math
import sys
import numpy as np

class DrivingGame:
    def __init__(self,
                 map_w=1000, map_h=1000,
                 screen_w=800, screen_h=600,
                 max_speed=4.0,
                 acc=0.2, brake=0.35, steer_angle=4.0,
                 fps=3000):
        
        pygame.init()
        self.MAP_W, self.MAP_H = map_w, map_h
        self.SCREEN_W, self.SCREEN_H = screen_w, screen_h
        self.MAX_SPEED = max_speed
        self.ACC = acc
        self.BRAKE = brake
        self.STEER_ANGLE = steer_angle
        self.FPS = fps

        self.ROAD_COLOR = (180, 180, 180)
        self.CHECK_COLOR = (0, 255, 200)
        self.OFFROAD_COLOR = (0, 0, 0)
        self.CAR_COLOR = (200, 0, 0)

        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        pygame.display.set_caption("Top-down Driving - Class")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Map surface
        self.map_surf = pygame.Surface((self.MAP_W, self.MAP_H))
        self._build_map()

        # Car properties
        self.car_w, self.car_h = 24, 12
        self.start_pos = (500, 90.0)
        self.reset()

        # Pre-render car base image
        self.base_car = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        pygame.draw.rect(self.base_car, self.CAR_COLOR, self.base_car.get_rect())

        self.externalkey_UP = False
        self.externalkey_DW = False
        self.externalkey_LEFT = False
        self.externalkey_RIGHT = False
        self.running = False
        
#----------------------------------------------------------------------------
    def _build_map(self, filepath="SliksGamePy/track.png"):
        """
        Carica la mappa da filepath (BMP, PNG, ecc.). L'immagine viene ridimensionata a MAP_W x MAP_H
        se necessario e assegnata a self.map_surf.
        Pixel con colore uguale a self.OFFROAD_COLOR sono considerati fuori strada.
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")

        # Carica immagine come Surface
        loaded = pygame.image.load(filepath)

        # Converti per velocità; mantiene canale alpha se presente
        if loaded.get_alpha() is not None:
            loaded = loaded.convert_alpha()
        else:
            loaded = loaded.convert()

        # Ridimensiona se la dimensione non corrisponde
        if loaded.get_width() != self.MAP_W or loaded.get_height() != self.MAP_H:
            loaded = pygame.transform.smoothscale(loaded, (self.MAP_W, self.MAP_H))

        # Assegna alla mappa
        self.map_surf = pygame.Surface((self.MAP_W, self.MAP_H))
        # Riempie prima col colore offroad per sicurezza
        self.map_surf.fill(self.OFFROAD_COLOR)
        # Copia l'immagine caricata sopra; le aree non-road dovrebbero essere disegnate già nell'immagine
        self.map_surf.blit(loaded, (0, 0))

#-----------------------------------------------------------------------------------------------
    def setAction(self, action):
        if action == 1:
               self.externalkey_UP = True
        if action == 2:
               self.externalkey_DW = True
        if action == 3:
               self.externalkey_LEFT = True
        if action == 4:
               self.externalkey_RIGHT = True
        if action == 5:
               self.externalkey_UP = True
               self.externalkey_LEFT = True
        if action == 6:
               self.externalkey_UP = True
               self.externalkey_RIGHT = True

#-----------------------------------------------------------------------------------------------
    def getAction_dim(self):
         return 8 

#-----------------------------------------------------------------------------------------------
    def getState(self) -> np.ndarray:
        state = []   
        rad = math.radians(self.car_angle)
        # Collision checks using a few sample points around the car
        self.state_points = [
            (self.car_x + math.cos(rad-0.1) * 110, self.car_y + math.sin(rad-0.1) * 110),
            (self.car_x + math.cos(rad+0.1) * 110, self.car_y + math.sin(rad+0.1) * 110),
            
            (self.car_x + math.cos(rad-0.4) * 60, self.car_y + math.sin(rad-0.4) * 60),
            (self.car_x + math.cos(rad-0.1) * 70, self.car_y + math.sin(rad-0.1) * 70),
            (self.car_x + math.cos(rad+0.1) * 70, self.car_y + math.sin(rad+0.1) * 70),
            (self.car_x + math.cos(rad+0.4) * 60, self.car_y + math.sin(rad+0.4) * 60),

            (self.car_x + math.cos(rad-1.0) * 25, self.car_y + math.sin(rad-1.0) * 25),
            (self.car_x + math.cos(rad-0.6) * 25, self.car_y + math.sin(rad-0.6) * 25),
            (self.car_x + math.cos(rad-0.2) * 30, self.car_y + math.sin(rad-0.2) * 30),
            (self.car_x + math.cos(rad+0.2) * 30, self.car_y + math.sin(rad+0.2) * 30),
            (self.car_x + math.cos(rad+0.6) * 25, self.car_y + math.sin(rad+0.6) * 25),
            (self.car_x + math.cos(rad+1.0) * 25, self.car_y + math.sin(rad+1.0) * 25),            
        ]

        state.append(self.speed)

        for px, py in self.state_points:
            if not self.is_on_road(px, py):
                state.append(1)
            else:
                state.append(-1)       

        return state        
    
#-----------------------------------------------------------------------------------------------
    def getState_dim(self):
        return 13 

#-----------------------------------------------------------------------------------------------
    def reset(self):
        # Reset car state
        self.car_x, self.car_y = float(self.start_pos[0]), float(self.start_pos[1])
        self.car_angle = 0.0
        self.speed = 0.0
        self.crashed = False
        self.step_count = 0
        self.state_points = []
        self.reward = 0

#-----------------------------------------------------------------------------------------------
    def rotate_center(self, image, angle):
        return pygame.transform.rotozoom(image, -angle, 1.0)

#-----------------------------------------------------------------------------------------------
    def is_on_road(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.MAP_W or iy >= self.MAP_H:
            return False
        return self.map_surf.get_at((ix, iy))[:3] != self.OFFROAD_COLOR

#-----------------------------------------------------------------------------------------------
    def step(self):
        self.step +=1
        dt = self.clock.tick(self.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
                        
        if not self.crashed:

            steer_angle = self.STEER_ANGLE

            # Acceleration: up/dw arrow
            if keys[pygame.K_a] or keys[pygame.K_UP] or self.externalkey_UP:
                self.speed += self.ACC
                steer_angle /= (self.speed +1)
            elif keys[pygame.K_z] or keys[pygame.K_DOWN] or self.externalkey_DW:
                self.speed -= self.BRAKE
                steer_angle /= (self.speed +1)
            else:
                self.speed -= 0.1

            # Steering: left/right arrows
            if keys[pygame.K_LEFT] or self.externalkey_LEFT:
                self.car_angle -= steer_angle

            if keys[pygame.K_RIGHT] or self.externalkey_RIGHT:
                self.car_angle += steer_angle

            #Reset keys
            self.externalkey_UP = False
            self.externalkey_DW = False
            self.externalkey_LEFT = False
            self.externalkey_RIGHT = False

            # Clamp speed
            self.speed = max(0.0, min(self.MAX_SPEED, self.speed))
   
            # Movement
            rad = math.radians(self.car_angle)
            dx = math.cos(rad) * self.speed
            dy = math.sin(rad) * self.speed
            self.car_x += dx
            self.car_y += dy
            self.distance = math.sqrt(dx*dx+dy*dy)

            # Collision checks using a few sample points around the car
            check_points = [
                (self.car_x, self.car_y),
                (self.car_x + math.cos(rad) * (self.car_w/2), self.car_y - math.sin(rad) * (self.car_w/2)),
                (self.car_x - math.cos(rad) * (self.car_w/2), self.car_y + math.sin(rad) * (self.car_w/2)),
                (self.car_x + math.cos(rad + math.pi/2) * (self.car_h/2), self.car_y - math.sin(rad + math.pi/2) * (self.car_h/2)),
                (self.car_x + math.cos(rad - math.pi/2) * (self.car_h/2), self.car_y - math.sin(rad - math.pi/2) * (self.car_h/2)),
            ]
            for px, py in check_points:
                if not self.is_on_road(px, py):
                    self.crashed = True
                    break

        # Camera centering and clamping
        cam_x = int(self.car_x - self.SCREEN_W / 2)
        cam_y = int(self.car_y - self.SCREEN_H / 2)
        cam_x = max(0, min(self.MAP_W - self.SCREEN_W, cam_x))
        cam_y = max(0, min(self.MAP_H - self.SCREEN_H, cam_y))

        # Draw map region
        self.screen.blit(self.map_surf, (0, 0), area=pygame.Rect(cam_x, cam_y, self.SCREEN_W, self.SCREEN_H))

        # Draw rotated car
        car_img = self.rotate_center(self.base_car, self.car_angle)
        car_rect = car_img.get_rect(center=(int(self.car_x - cam_x), int(self.car_y - cam_y)))
        self.screen.blit(car_img, car_rect)        

        #  STATE_PT Converti alle coord relative alla camera e disegna solo 3 punti (es. i primi 3)
        for px, py in self.state_points:
            screen_x = int(px - cam_x)
            screen_y = int(py - cam_y)
            if self.is_on_road(px, py):
                col = (128, 128, 128)
            else:
                col = (255, 128, 128)
            pygame.draw.circle(self.screen, col, (screen_x, screen_y), 4)  # rosso, raggio 4

        # opzionale: disegna etichette dei punti
        font = pygame.font.get_default_font()
        f = pygame.font.Font(font, 14)
        for i, (px, py) in enumerate(self.state_points):
            sx, sy = int(px - cam_x), int(py - cam_y)
            lbl = f.render(str(i), True, (128, 128, 128) )
            self.screen.blit(lbl, (sx+6, sy-6))


        # Crash overlay and reset handling
        if self.crashed:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))
            text = self.font.render("CRASHED! Press R to reset.", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.reset()

        # HUD
        hud = self.font.render(f"Speed: {self.speed:.2f} px/frame  Pos: {int(self.car_x)},{int(self.car_y)}", True, (0, 255, 255))
        self.screen.blit(hud, (10, self.SCREEN_H - 30))

        pygame.display.flip()
        self.step_count += 1
        next_state = self.getState()

        self.reward = self.speed - 1
        if (self.crashed):
            reward = -10
        
        done = self.crashed
        info =  self.reward
        return next_state,  self.reward, done, info

#-----------------------------------------------------------------------------------------------
    def run(self):
        self.running = True
        while self.running:
            self.step()
        self.quit()

#-----------------------------------------------------------------------------------------------
    def init_run(self):
        self.running = True
#-----------------------------------------------------------------------------------------------
    def step_run(self, action):
        if self.running:
            self.setAction(action)
            return self.step()
        return self.quit()
#-----------------------------------------------------------------------------------------------
    def quit(self):
        pygame.quit()
        sys.exit()

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    game = DrivingGame()
    
    game.init_run()
    action = 0
    while game.running:
        state = game.getState()
        score = game.step_run(action)

