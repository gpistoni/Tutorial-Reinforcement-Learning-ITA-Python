import pygame
import math
import sys
import numpy as np

class DrivingGame:
    def __init__(self,
                 fileMap,
                 max_speed,
                 render_decmation = 1,
                 screen_w=1000, screen_h=1000,
                 acc=0.2, brake=0.35, steer_angle=4.0,
                 fps=30):
        
        pygame.init()
        self.SCREEN_W, self.SCREEN_H = screen_w, screen_h
        self.MIN_SPEED = 0.2
        self.max_speed = max_speed
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
        self.lap_time = pygame.time.get_ticks()

        # Map surface
        self._build_map(fileMap)

        # Car properties
        self.car_w, self.car_h = 24, 12
        self.start_pos = (500, 120.0)
        self.reset()

        # Pre-render car base image
        self.base_car = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        pygame.draw.rect(self.base_car, self.CAR_COLOR, self.base_car.get_rect())

        self.externalkey_UP = False
        self.externalkey_DW = False
        self.externalkey_LEFT = False
        self.externalkey_RIGHT = False
        self.running = False
        self.render_decmation = render_decmation
        self.tryMode = 0
        
#----------------------------------------------------------------------------
    def _build_map(self, fileMap):
        """
        Carica la mappa da filepath (BMP, PNG, ecc.). 
        Pixel con colore uguale a self.OFFROAD_COLOR sono considerati fuori strada.
        """
        import os
        if not os.path.exists(fileMap):
            raise FileNotFoundError(f"Map file not found: {fileMap}")

        # Carica immagine come Surface
        loaded = pygame.image.load(fileMap)

        # Converti per velocità; mantiene canale alpha se presente
        if loaded.get_alpha() is not None:
            loaded = loaded.convert_alpha()
        else:
            loaded = loaded.convert()

        # Ridimensiona se la dimensione non corrisponde
        self.MAP_W = loaded.get_width()
        self.MAP_H = loaded.get_height()            

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
        self.state_points = []

########################################
        self.dir_points = []
        for an in range(10):
            self.dir_points.append(1)
            realan = rad + 2*(an-5)
            rcos = math.cos(realan)
            rsin = math.sin(realan)
            for d in range(100):
                ptx, pty = (self.car_x + rcos * d, self.car_y + rsin * d)
                if not self.is_on_road(ptx, pty):
                    self.dir_points[an] = d/100
                    break

########################################
        state.append(self.speed/10)
        state.append(self.derive/100)

        for e in self.dir_points:
            state.append(e)

        return state        
    
#-----------------------------------------------------------------------------------------------
    def getState_dim(self):
        return 12

#-----------------------------------------------------------------------------------------------
    def reset(self):
        # Reset car state
        self.car_x, self.car_y = float(self.start_pos[0]), float(self.start_pos[1])
        self.car_angle = 0.0
        self.wheel_angle = 0.0
        self.speed = 0.0
        self.derive = 0.0
        self.crashed = False
        self.step_count = 0
        self.state_points = []
        self.reward = 0
        self.distance = 0 
        self.dist_ckeckpoint = []
        self.dist_ckeckpoint.append((self.car_x, self.car_y))
        self.lap_time = pygame.time.get_ticks()

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
    def rescale_to_minus1_1(self, val: float, val_min: float, val_max: float) -> float:
        return ((val - val_min) / (val_max - val_min)) * 2.0 - 1.0

#-----------------------------------------------------------------------------------------------
    def step(self):
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
            elif keys[pygame.K_z] or keys[pygame.K_DOWN] or self.externalkey_DW:
                self.speed -= self.BRAKE
            else:
                self.speed -= 0.1
            self.speed = max(self.speed,self.MIN_SPEED)

            # Steering: left/right arrows
            if keys[pygame.K_LEFT] or self.externalkey_LEFT:
                self.car_angle -= steer_angle

            if keys[pygame.K_RIGHT] or self.externalkey_RIGHT:
                self.car_angle += steer_angle
            
            #Slip
            self.derive = self.car_angle - self.wheel_angle
            self.derive = min( self.derive, 90 )
            self.derive = max( self.derive, -90 )

            Act_derive = min( self.derive, steer_angle )
            Act_derive = max( Act_derive, -steer_angle )
            if ( self.speed > 0 ):
                self.wheel_angle += Act_derive / (self.speed/6)
            self.speed -=  self.speed * abs( self.derive / 200.0)

            #Reset keys
            self.externalkey_UP = False
            self.externalkey_DW = False
            self.externalkey_LEFT = False
            self.externalkey_RIGHT = False

            # Clamp speed
            self.speed = max(0.0, min(self.max_speed, self.speed))
   
            # Movement
            radw = math.radians(self.wheel_angle)
            dx = math.cos(radw) * self.speed
            dy = math.sin(radw) * self.speed
            self.car_x += dx
            self.car_y += dy
            self.distance += math.sqrt(dx*dx+dy*dy)

            chkp_dist = min(math.dist((self.car_x, self.car_y), cp) for cp in self.dist_ckeckpoint[-10:]   )
            if (chkp_dist > 100):
                self.dist_ckeckpoint.append((self.car_x, self.car_y))

            # Collision checks using a few sample points around the car
            rad = math.radians(self.car_angle)
            check_points = [
                (self.car_x, self.car_y),
                (self.car_x + math.cos(rad) * (self.car_w/2), self.car_y - math.sin(rad) * (self.car_w/2)),
                (self.car_x - math.cos(rad) * (self.car_w/2), self.car_y + math.sin(rad) * (self.car_w/2)),
                (self.car_x + math.cos(rad + math.pi/2) * (self.car_h/2), self.car_y - math.sin(rad + math.pi/2) * (self.car_h/2)),
                (self.car_x + math.cos(rad - math.pi/2) * (self.car_h/2), self.car_y - math.sin(rad - math.pi/2) * (self.car_h/2)),
            ]
            out_of_road = 0
            for px, py in check_points:
                if not self.is_on_road(px, py):
                    out_of_road += 1                    
            if (out_of_road >1):
                self.crashed = True

            #Lap Check
            d = math.dist((self.car_x, self.car_y), self.start_pos)
            lt = pygame.time.get_ticks() - self.lap_time
            if d < 40 and lt > 2000:
                #Lap
                print(f"LapTime s: {lt/1000.0}")
                self.lap_time = pygame.time.get_ticks()
                                

        # Camera centering and clamping
        cam_x = int(self.car_x - self.SCREEN_W / 2)
        cam_y = int(self.car_y - self.SCREEN_H / 2)
        cam_x = max(0, min(self.MAP_W - self.SCREEN_W, cam_x))
        cam_y = max(0, min(self.MAP_H - self.SCREEN_H, cam_y))

        if(self.step_count% self.render_decmation==0):
            # Draw map region
            self.screen.blit(self.map_surf, (0, 0), area=pygame.Rect(cam_x, cam_y, self.SCREEN_W, self.SCREEN_H))

            # Draw rotated car
            car_img = self.rotate_center(self.base_car, self.car_angle)
            car_rect = car_img.get_rect(center=(int(self.car_x - cam_x), int(self.car_y - cam_y)))
            self.screen.blit(car_img, car_rect)        

            #  STATE_PT Converti alle coord relative alla camera e disegna solo 3 punti (es. i primi 3)
            if False:
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

            #Chechpoints
            for px, py in self.dist_ckeckpoint[-10:]:
                screen_x = int(px - cam_x)
                screen_y = int(py - cam_y)
                pygame.draw.circle(self.screen,  self.OFFROAD_COLOR, (screen_x, screen_y), 12)
     
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
                    self.tryMode=1
  
            # HUD
            hud = self.font.render(f"Speed: {self.speed:.2f} px/frame  Pos: {int(self.car_x)},{int(self.car_y)} Dist: {int(self.distance)} Chk:{self.dist_ckeckpoint.__len__()}", True, (128, 128, 255))
            self.screen.blit(hud, (10, self.SCREEN_H - 30))
            pygame.display.flip()
        
        self.step_count += 1
        next_state = self.getState()

        #Reward speed - derive 50% distance
        self.reward = self.rescale_to_minus1_1(self.speed, 0, self.max_speed)
        #self.reward -= abs(self.derive) / 100                   #200 influisce sulla % di impatto reward
        self.reward += self.dist_ckeckpoint.__len__() / 200
        if (self.crashed):
            self.reward = -10
        
        done = self.crashed
        info =  self.distance
        return next_state, self.reward, done, info

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
    game = DrivingGame( fileMap="SliksGamePy/track_0.png", max_speed=5 )
    
    game.init_run()
    action = 0
    while game.running:
        state = game.getState()
        score = game.step_run(action)

