export const vertexShader = /* glsl */`
  out vec2 screenUV;
  void main() {
    screenUV = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const fragmentShader = /* glsl */`
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D clouds;
  uniform sampler2D colorBuffer;
  uniform sampler2D depthBuffer;
  uniform mat4 inverseProjection;
  uniform mat4 cameraWorld;
  uniform vec3 cloudMin;
  uniform vec3 cloudMax;
  uniform vec3 flightFocus;
  uniform float passage;
  in vec2 screenUV;
  out vec4 finalColor;

  vec3 toBlender(vec3 p) { return vec3(p.x, -p.z, p.y); }
  vec3 linearRGB(vec3 color) { return pow(color, vec3(2.2)); }

  vec3 skyColor(vec3 ray) {
    float height = smoothstep(-0.28, 0.74, ray.z);
    vec3 sky = mix(linearRGB(vec3(0.82, 0.73, 0.86)), linearRGB(vec3(0.42, 0.56, 0.89)), height);
    vec3 sun = normalize(vec3(0.20, 0.86, 0.47));
    float alignment = max(dot(sun, ray), 0.0);
    sky = mix(sky, linearRGB(vec3(1.0, 0.80, 0.58)), 0.64 * pow(alignment, 47.0));
    return mix(sky, linearRGB(vec3(1.0, 0.93, 0.78)), smoothstep(0.9984, 0.9999, alignment));
  }

  vec3 cloudColor(float light, float distance, vec3 background) {
    vec3 shadow = linearRGB(vec3(0.55, 0.46, 0.66));
    vec3 rose = linearRGB(vec3(0.94, 0.62, 0.68));
    vec3 peach = linearRGB(vec3(1.0, 0.86, 0.68));
    vec3 color = mix(shadow, rose, smoothstep(0.04, 0.46, light));
    color = mix(color, peach, smoothstep(0.41, 0.97, light));
    return mix(color, background, 0.54*smoothstep(55.0, 530.0, distance));
  }

  void main() {
    vec2 ndc = screenUV*2.0 - 1.0;
    float depth = texture(depthBuffer, screenUV).r;
    vec4 plane = inverseProjection * vec4(ndc, depth*2.0 - 1.0, 1.0);
    vec3 pointInView = plane.xyz / plane.w;
    vec3 ray = toBlender(normalize(mat3(cameraWorld) * pointInView));
    vec3 eye = toBlender(cameraWorld[3].xyz);
    vec3 sky = skyColor(ray);
    bool solid = depth < 0.999999;
    vec3 behind = solid ? texture(colorBuffer, screenUV).rgb : sky;

    vec3 small = vec3(0.00001);
    vec3 inverseRay = mix(-1.0/max(abs(ray), small), 1.0/max(abs(ray), small), step(vec3(0.0), ray));
    vec3 a = (cloudMin - eye)*inverseRay;
    vec3 b = (cloudMax - eye)*inverseRay;
    vec3 nearAxis = min(a, b);
    vec3 farAxis = max(a, b);
    float near = max(0.0, max(max(nearAxis.x, nearAxis.y), nearAxis.z));
    float far = min(min(farAxis.x, farAxis.y), farAxis.z);
    if (solid) far = min(far, length(pointInView));

    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)))*43758.5453);
    float distance = near + 0.65*jitter;
    for (int i = 0; i < 112; i++) {
      if (distance >= far || transmittance < 0.015) break;
      float stride = min(far - distance, mix(1.45, 7.0, smoothstep(6.0, 360.0, distance)));
      vec3 position = eye + ray*(distance + stride*0.42);
      vec2 medium = texture(clouds, (position - cloudMin)/(cloudMax - cloudMin)).rg;
      if (medium.r > 0.005) {
        float density = medium.r;
        if (passage > 0.0) {
          // Let the nearby cloud read as mist during a crossing. Distant pillars stay dense.
          float nearWhale = 1.0 - smoothstep(20.0, 51.0, length(position - flightFocus));
          float nearEye = 0.82*(1.0 - smoothstep(8.0, 25.0, distance));
          density *= mix(1.0, 0.11, passage*max(nearWhale, nearEye));
        }
        float share = 1.0 - exp(-density*stride*0.64);
        float illumination = medium.g;
        accumulated += transmittance*share*cloudColor(illumination, distance, sky);
        transmittance *= 1.0 - share;
      }
      distance += stride;
    }
    vec3 display = accumulated + transmittance*behind;
    finalColor = vec4(pow(clamp(display, 0.0, 1.0), vec3(1.0/2.2)), 1.0);
  }
`;
