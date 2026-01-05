import pygame
import math
import sys

class DrivingGame:
    def __init__(self,
                 map_w=1000, map_h=1000,
                 screen_w=800, screen_h=600,
                 max_speed=2.0,
                 acc=0.1, brake=0.35, steer_angle=4.0,
                 fps=60):
        
        pygame.init()
        self.MAP_W, self.MAP_H = map_w, map_h
        self.SCREEN_W, self.SCREEN_H = screen_w, screen_h
        self.MAX_SPEED = max_speed
        self.ACC = acc
        self.BRAKE = brake
        self.STEER_ANGLE = steer_angle
        self.FPS = fps

        self.ROAD_COLOR = (180, 180, 180)
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
        self.start_pos = (500, 75.0)
        self.reset()

        # Pre-render car base image
        self.base_car = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        pygame.draw.rect(self.base_car, self.CAR_COLOR, self.base_car.get_rect())

        self.externalkey_UP = False
        self.running = False

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

    def setAction(self, action):
        if action == 1:
               self.externalkey_UP = True

    def getstate(self):
        state = []        
        state.append(1)
        return state
        

    def reset(self):
        # Reset car state
        self.car_x, self.car_y = float(self.start_pos[0]), float(self.start_pos[1])
        self.car_angle = 0.0
        self.speed = 0.0
        self.score = 0.0
        self.crashed = False
        self.step_count = 0

    def rotate_center(self, image, angle):
        return pygame.transform.rotozoom(image, -angle, 1.0)

    def is_on_road(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.MAP_W or iy >= self.MAP_H:
            return False
        return self.map_surf.get_at((ix, iy))[:3] != self.OFFROAD_COLOR

    def step(self):
        dt = self.clock.tick(self.FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        
                
        if not self.crashed:

            # Steering: keypad 4/6 or left/right arrows
            if keys[pygame.K_KP4] or keys[pygame.K_LEFT]:
                self.car_angle += self.STEER_ANGLE / (self.speed + 0.1)

            if keys[pygame.K_KP6] or keys[pygame.K_RIGHT]:
                self.car_angle -= self.STEER_ANGLE / (self.speed + 0.1)

            # Acceleration with keypad 8 or up arrow
            if keys[pygame.K_KP8] or keys[pygame.K_UP] or self.externalkey_UP:
                self.speed += self.ACC
            else:
                self.speed -= self.BRAKE

            # Clamp speed
            self.speed = max(0.0, min(self.MAX_SPEED, self.speed))
            self.score += self.speed - 1

            # Movement
            rad = math.radians(self.car_angle)
            dx = math.cos(rad) * self.speed
            dy = math.sin(rad) * self.speed
            self.car_x += dx
            self.car_y += dy

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
        hud = self.font.render(f"Speed: {self.speed:.2f} px/frame Score:{self.score:.2f}  Pos: {int(self.car_x)},{int(self.car_y)}", True, (0, 255, 255))
        self.screen.blit(hud, (10, self.SCREEN_H - 30))

        pygame.display.flip()
        self.step_count += 1
        return self.score

    def run(self):
        self.running = True
        while self.running:
            self.step()
        self.quit()

    def init_run(self):
        self.running = True

    def step_run(self, action):
        if self.running:
            self.setAction(action)
            return self.step()
        return self.quit()

    def quit(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = DrivingGame()
    
    game.init_run()
    action = 1
    while game.running:
        state = game.getstate()
        score = game.step_run(action)

