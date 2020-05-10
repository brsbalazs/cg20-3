//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 20;
static bool isxPressed = false;
static bool isXPressed = false;
static bool isyPressed = false;
static bool isYPressed = false;
static bool iszPressed = false;
static bool isZPressed = false;
static float dT = 0.0f;



//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extinsic: szem poz, lookat, preferalt fuggoleges irany
    float fov, asp, fp, bp;        // intrinsic: fuggoleges latoszog, aspektusarany, elso vagosik, hatso vagosik
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 10;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }

    void Animate(float t) { }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka; //diff, spek, ambiens visszaverodesi kepesseg
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le; //ambiens es direkcionalis fenykibocsajtasi mod/mertek
    vec4 wLightPos; //fenyforras helye - ha a 4. koord 1, akkor a vegtelenben van (=direkcionalis) - így egyszerre kezelheto pozicionalis es direktcionalis fenyforrasokat is


    void Animate(float t) {    }
};

//---------------------------
class CheckerBoardTexture : public Texture { //textura tipus implementacioja, letrehozasa
//---------------------------
public:
    CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
            image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
        }
        create(width, height, image, GL_NEAREST);
    }
};

//---------------------------
class StripyTexture : public Texture { //textura tipus implementacioja, letrehozasa
//---------------------------
public:
    StripyTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
        for (int x = 0; x < width; x++)
        {
            int counter = 0;
            bool isBlue = true;
            
            for (int y = 0; y < height; y++) {
                
                if(counter % 5 == 0)
                {
                    isBlue = !isBlue;
                    counter = 0;
                }
                
                if(isBlue) image[y * width + x] = blue;
                else{image[y * width + x] = yellow; }
                counter ++;
            }
        }
            
        create(width, height, image, GL_NEAREST);
    }
};

//---------------------------
class RainbowTexture : public Texture { //textura tipus implementacioja, letrehozasa
//---------------------------
public:
    RainbowTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
        for (int x = 0; x < width; x++)
        {
          
            for (int y = 0; y < height; y++) {
                image[y * width + x] = vec4(x/100,y/100,x/100,1);
            }
        }
            
        create(width, height, image, GL_NEAREST);
    }
};


//---------------------------
//Egyfajta interface az objektumok es shaderek kozott
//A render statebe pakolja be az objektum az osszes olyan valtozojat, amit a shaderben is szeretne ervenyesiteni
struct RenderState {
//---------------------------
    mat4               MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    Texture *          texture;
    vec3               wEye;
};


//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
//per vertex arnyalas
class GouraudShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };
        
        struct Material {
            vec3 kd, ks, ka;
            float shininess;
        };

        uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
        uniform Light[8] lights;     // light source direction
        uniform int   nLights;         // number of light sources
        uniform vec3  wEye;          // pos of eye
        uniform Material  material;  // diffuse, specular, ambient ref

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space

        out vec3 radiance;            // reflected radiance

        void main() {
            gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
            // radiance computation
            vec4 wPos = vec4(vtxPos, 1) * M;
            vec3 V = normalize(wEye * wPos.w - wPos.xyz);
            vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
            if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein

            radiance = vec3(0, 0, 0);
            for(int i = 0; i < nLights; i++) {
                vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
                vec3 H = normalize(L + V);
                float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
            }
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        in  vec3 radiance;      // interpolated radiance
        out vec4 fragmentColor; // output goes to frame buffer

        void main() {
            fragmentColor = vec4(radiance, 1);
        }
    )";
    
    //Per vertex arnyalas bind fuggvenye
public:
    GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };

        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform Light[8] lights;    // light sources
        uniform int   nLights;
        uniform vec3  wEye;         // pos of eye

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
        layout(location = 2) in vec2  vtxUV;

        out vec3 wNormal;            // normal in world space
        out vec3 wView;             // view in world space
        out vec3 wLight[8];            // light dir in world space
        out vec2 texcoord;

        void main() {
            gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
            // vectors for radiance computation
            vec4 wPos = vec4(vtxPos, 1) * M;
            for(int i = 0; i < nLights; i++) {
                wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
            }
            wView  = wEye * wPos.w - wPos.xyz;
            wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
            texcoord = vtxUV;
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };

        struct Material {
            vec3 kd, ks, ka;
            float shininess;
        };

        uniform Material material;
        uniform Light[8] lights;    // light sources
        uniform int   nLights;
        uniform sampler2D diffuseTexture;

        in  vec3 wNormal;       // interpolated world sp normal
        in  vec3 wView;         // interpolated world sp view
        in  vec3 wLight[8];     // interpolated world sp illum dir
        in  vec2 texcoord;
        
        out vec4 fragmentColor; // output goes to frame buffer

        void main() {
            vec3 N = normalize(wNormal);
            vec3 V = normalize(wView);
            if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
            vec3 texColor = texture(diffuseTexture, texcoord).rgb;
            vec3 ka = material.ka * texColor;
            vec3 kd = material.kd * texColor;

            vec3 radiance = vec3(0, 0, 0);
            for(int i = 0; i < nLights; i++) {
                vec3 L = normalize(wLight[i]);
                vec3 H = normalize(L + V);
                float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                // kd and ka are modulated by the texture
                radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
            }
            fragmentColor = vec4(radiance, 1);
        }
    )";
    
     
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

//---------------------------
//festest szimulalo shader
class NPRShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform    vec4  wLightPos;
        uniform vec3  wEye;         // pos of eye

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
        layout(location = 2) in vec2  vtxUV;

        out vec3 wNormal, wView, wLight;                // in world space
        out vec2 texcoord;

        void main() {
           gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
           vec4 wPos = vec4(vtxPos, 1) * M;
           wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
           wView  = wEye * wPos.w - wPos.xyz;
           wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
           texcoord = vtxUV;
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        uniform sampler2D diffuseTexture;

        in  vec3 wNormal, wView, wLight;    // interpolated
        in  vec2 texcoord;
        out vec4 fragmentColor;                // output goes to frame buffer

        void main() {
           vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
           float y = (dot(N, L) > 0.5) ? 1 : 0.5;
           if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
           else                         fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
        }
    )";
public:
    NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniform(state.lights[0].wLightPos, "wLightPos");
    }
};


//---------------------------
struct VertexData {
//---------------------------
    vec3 position, normal;
    vec2 texcoord;
};

//---------------------------
//Geometriai alakok alaposztalya
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao); //minden vaohoz egy vbo, amibe omlesztve lesznek az adatok: csupont helye, felulet normalvektora, textura koordinatak
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

//---------------------------
//Parametrikus felulet alaposztaly
class ParamSurface : public Geometry {
//---------------------------

public:
    
    unsigned int nVtxPerStrip, nStrips; // egy stripen belul osszesen hany csucspontot adunk meg, hany sora van az adott felbontasnak
       std::vector<VertexData> vtxData;
       bool isTetrahedron = false;
    
    ParamSurface() { nVtxPerStrip = nStrips = 0; }
    
    virtual VertexData GenVertexData(float u, float v) = 0;
    
    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        
        if(!isTetrahedron)
        {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= M; j++) {
                    vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                    vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
                }
            }
            
            glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
            

        }
        else if(isTetrahedron)
        {
            GenVertexData(0, 0);
            glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

        }
        
        
        //glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

//---------------------------
struct Clifford {
//---------------------------
    float f, d;
    Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
    Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
    Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
    Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
    Clifford operator/(Clifford r) {
        float l = r.f * r.f;
        return (*this) * Clifford(r.f / l, -r.d / l);
    }
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g)/Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
//saját cliffor metódusok
Clifford Cosh(Clifford g) { return Clifford(cosh(g.f),sinh(g.f) * g.d); }
Clifford Sinh(Clifford g) { return Clifford(sinh(g.f),cosh(g.f) * g.d); }



//---------------------------
//Dual-numbers class
template<class T> struct Dnum{
//---------------------------
    float f;
    T d;
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f); }
    Dnum operator/(Dnum r) { return Dnum( f / r.f, (r.f * d - r.d * f) / r.f / r.f);}
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sin(g.f), cos(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cos(g.f), -sin(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g)/Cos(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), 1 / g.f * g.d); }
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
//saját cliffor metódusok
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f),sinh(g.f) * g.d); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f),cosh(g.f) * g.d); }

typedef Dnum<vec2> Dnum2;

//---------------------------
class Sphere : public ParamSurface {
//---------------------------
    
    float radius = 1.2f;
public:
    Sphere() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vd.normal = vec3(radius * cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * cosf(v * (float)M_PI));
        vd.texcoord = vec2(u, v);
        return vd;
    }
    
    
};

//---------------------------
class VirusParent : public ParamSurface {
//---------------------------
    float radius = 1.2f;
public:
    VirusParent() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        radius = (1.3f + (sin((19.0f*u) + (24.0f*v)))/8);
        
        //TODO: determining normal of the surface
        vd.position = vd.normal = vec3( radius * cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * cosf(v * (float)M_PI));
        vd.texcoord = vec2(u, v);
        return vd;
    }
    
    VertexData GenVertexDataForTime(float u, float v, float tend) {
        VertexData vd;
        radius = 1.3f + ((sin((19.0f*u) + (24.0f*v)))/8 * cosf(tend*2));
        
        //TODO: determining normal of the surface
        vd.position = vd.normal = vec3( radius * cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       radius * cosf(v * (float)M_PI));
        vd.texcoord = vec2(u, v);
        return vd;
    }
    
    //virus waving movement
    void reCreate(int N = tessellationLevel, int M = tessellationLevel, float tend = 0) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        
        //vao letrehozas bindolas
        glGenVertexArrays(1, &vao); //minden vaohoz egy vbo, amibe omlesztve lesznek az adatok: csupont helye, felulet normalvektora, textura koordinatak
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        while(vtxData.size() != 0){ vtxData.pop_back();} // vertices on the CPU
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexDataForTime((float)j / M, (float)i / N, tend));
                vtxData.push_back(GenVertexDataForTime((float)j / M, (float)(i + 1) / N, tend));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }
};

//---------------------------
class Tetrahedron : public ParamSurface {
//---------------------------
    /*A tetrtaéder definíciója a 3 alappontból, plusz magasságból
     -felületi normális: a 4pont átlagából a vizsgált pontba húzott vektor normalizált alakja*/
    
    float radius = 1.2f;
    vec3 vert1;
    vec3 vert2;
    vec3 vert3;
    vec3 vert4;
    vec3 orientation;
    float height;
    
public:
    
    
    
    //orientation: egy egysegvektor ami megmutatja, hogy az alaptol milyen iranyba kell mennunk magassagnyival,
    // megadhato neki a parent surface normalvektora pl.
    Tetrahedron(vec3 _vert1, vec3 _vert2, vec3 _vert3, float _height, vec3 _orientation)
    {
        isTetrahedron = true;
        vert1 = _vert1;
        vert2 = _vert2;
        vert3 = _vert3;
        orientation = _orientation;
        height = _height;
        create();
    }
    
    vec3 getCenter (){
        return vec3((vert1+vert2+vert3+vert4)/4.0f);
    }

    VertexData GenVertexData(float u, float v) {
        
        //the base of the triangle is 1,2,3
        vec3 centerOfBaseSurface = (vert1 + vert2 + vert3)/3.0f;
        vert4 = centerOfBaseSurface + (orientation * height);
        vec3 centerOfTetrahedron = (centerOfBaseSurface + vert4) / 2.0f;
        
        //collecting sides
        
        VertexData vd1;
        VertexData vd2;
        VertexData vd3;
        VertexData vd4;
        VertexData vd5;
        VertexData vd6;

        //TODO: a normalvektor kicsit rossz iranyba mutat, csak az alapnal teljesen jo, mindig a negyedik csucsbol kene a kozeppontba huzni

        vd1.position = vert1;
        vd1.normal = normalize(centerOfBaseSurface-centerOfTetrahedron);
        vd1.texcoord = vec2(vert1.x,vert1.y);
        
        vd2.position = vert2;
        vd2.normal = normalize(centerOfBaseSurface-centerOfTetrahedron);
        vd2.texcoord = vec2(vert2.x,vert2.y);
        
        vd3.position = vert3;
        vd3.normal = normalize(centerOfBaseSurface-vert4);
        vd3.texcoord = vec2(vert3.x,vert3.y);
        
        //triangle of 2,3,4
        vd4.position = vert4;
        vd4.normal = normalize(((vert2+vert3+vert4)/3.0f)-vert1);
        vd4.texcoord = vec2(vert4.x,vert4.y);
        
        //triangle of 3,4,1
        vd5.position = vert1;
        vd5.normal = normalize(((vert3+vert4+vert1)/3.0f)-vert2);
        vd5.texcoord = vec2(vert1.x,vert4.y);
        
        //triangle of 1,4,2
        vd6.position = vert2;
        vd6.normal = normalize(((vert1+vert4+vert2)/3.0f)-vert3);
        vd6.texcoord = vec2(vert2.x,vert2.y);
        
        vtxData.push_back(vd1);
        vtxData.push_back(vd2);
        vtxData.push_back(vd3);
        vtxData.push_back(vd4);
        vtxData.push_back(vd5);
        vtxData.push_back(vd6);

        VertexData retValue;
        return retValue;
    }
    
    void GenVertexDataForTime(float u, float v, float tend) {
        //the base of the triangle is 1,2,3
        vec3 centerOfBaseSurface = (vert1 + vert2 + vert3)/3.0f;
        vert4 = centerOfBaseSurface + (orientation * (1.0f + (height * fabs(cos(tend)))));
        vec3 centerOfTetrahedron = (centerOfBaseSurface + vert4) / 2.0f;
        
        //collecting sides
        
        VertexData vd1;
        VertexData vd2;
        VertexData vd3;
        VertexData vd4;
        VertexData vd5;
        VertexData vd6;


        vd1.position = vert1;
        vd1.normal = normalize(centerOfBaseSurface-centerOfTetrahedron);
        vd1.texcoord = vec2(vert1.x,vert1.y);
        
        vd2.position = vert2;
        vd2.normal = normalize(centerOfBaseSurface-centerOfTetrahedron);
        vd2.texcoord = vec2(vert2.x,vert2.y);
        
        vd3.position = vert3;
        vd3.normal = normalize(centerOfBaseSurface-vert4);
        vd3.texcoord = vec2(vert3.x,vert3.y);
        
        //triangle of 2,3,4
        vd4.position = vert4;
        vd4.normal = normalize(((vert2+vert3+vert4)/3.0f)-vert1);
        vd4.texcoord = vec2(vert4.x,vert4.y);
        
        //triangle of 3,4,1
        vd5.position = vert1;
        vd5.normal = normalize(((vert3+vert4+vert1)/3.0f)-vert2);
        vd5.texcoord = vec2(vert1.x,vert4.y);
        
        //triangle of 1,4,2
        vd6.position = vert2;
        vd6.normal = normalize(((vert1+vert4+vert2)/3.0f)-vert3);
        vd6.texcoord = vec2(vert2.x,vert2.y);
        
        vtxData.push_back(vd1);
        vtxData.push_back(vd2);
        vtxData.push_back(vd3);
        vtxData.push_back(vd4);
        vtxData.push_back(vd5);
        vtxData.push_back(vd6);

        
        return;
    }
    
    //antibody waving movement
    void reCreate(int N = tessellationLevel, int M = tessellationLevel, float tend = 0) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        
        //vao letrehozas bindolas
        glGenVertexArrays(1, &vao); //minden vaohoz egy vbo, amibe omlesztve lesznek az adatok: csupont helye, felulet normalvektora, textura koordinatak
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        while(vtxData.size() != 0){ vtxData.pop_back();} // vertices on the CPU
        
        GenVertexDataForTime(0, 0, tend);
        
        glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }
};




//---------------------------
//saját tractricoid implementáció
class Tractricoid : public ParamSurface{
//---------------------------
    //TODO: magasság még konfigurálandó
    float height = 3.0f;
public:
  Tractricoid() { create(); }

  VertexData GenVertexData(float u, float v) {
      VertexData vd;
      vd.texcoord = vec2(u, v);
      Dnum2 X, Y, Z;
      Dnum2 U(u, vec2(1,0)), V (v, vec2(0,1));
      eval(U, V, X, Y, Z);
      
      vd.position = vec3(X.f, Y.f, Z.f);
      vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
      vd.normal = cross(drdU, drdV);
      return vd;
  }
    
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z){
        U = U * height;
        V = V * 2 * M_PI;
        X = Cos(V) / Cosh(U);
        Y = Sin(V) / Cosh(U);
        Z = U - (Sinh(U)/Cosh(U)); // U - Tanh(U)
    }
};


//---------------------------
struct Object {
//---------------------------
    Shader *   shader; //pixelarnyalo program ami eppen akkor el amikor ezen objektumhoz rendelt csucspontok vegigmennek a pipelineon
    Material * material;
    Texture *  texture;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
public:
    Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }
    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    virtual void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    //az animaciot biztosito fuggveny
    virtual void Animate(float tstart, float tend)
    {
        rotationAngle = 0.8f * tend; //saját tengely körüli forgás
    }
};

//---------------------------
struct VirusObject : Object{
//---------------------------
    VirusParent* virusParent; //transformed sphere body of the virus
public:
    
    //children
    std::vector<Tractricoid*> children;

    //constr
    VirusObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry, VirusParent * _parent) :
    Object(_shader, _material, _texture, _geometry){
        virusParent = _parent;
    }
    
    void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    //az animaciot biztosito fuggveny
    //TODO: animacio: mozgas, hullamzas, talan forgas is
     void Animate(float tstart, float tend)
    {
        /*
        rotationAngle = 0.8f * tend; //saját tengely körüli forgás
        vec3 translationVec = vec3(sinf(tend/2.0f), sinf(tend/3.0f), sinf(tend/5.0f));
        translationVec = normalize(translationVec);
        translation = translationVec;
        rotationAxis = cosf(tend);
        virusParent->reCreate(tessellationLevel, tessellationLevel, tend);*/
        
        rotationAngle = cosf(tend); //saját tengely körüli forgás
        vec3 translationVec = vec3(sinf(tend/2.0f), sinf(tend/3.0f), sinf(tend/5.0f));
        translationVec = normalize(translationVec);
        translation = translationVec;
        rotationAxis = normalize(vec3(sinf(tend/2.0f), sinf(tend/3.0f), sinf(tend/5.0f)));
        virusParent->reCreate(tessellationLevel, tessellationLevel, tend);
    }
    
    //TODO: collection tractricoid objects aroud the sphere
    void addChildren(){}
};

//---------------------------
struct AntibodyObject : Object{
//---------------------------
    Tetrahedron * antibodyParent; //transformed sphere body of the virus
    std::vector<Tetrahedron*> childrenDepth1;
    std::vector<Tetrahedron*> childrenDepth2;
    std::vector<Tetrahedron*> childrenDepth3;
    float childPath1Height = 0.9f;
    float childPath2Height = 0.5f;
    float childPath3Height = 0.2f;

    
public:

    //constr
    AntibodyObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry, Tetrahedron * _parent) :
    Object(_shader, _material, _texture, _geometry){antibodyParent = _parent;}
    
    void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        for(int i = 0; i < childrenDepth1.size(); i++)
        {
            childrenDepth1[i]->Draw();
        }
        for(int i = 0; i < childrenDepth2.size(); i++)
        {
            childrenDepth2[i]->Draw();
        }
        for(int i = 0; i < childrenDepth3.size(); i++)
        {
            childrenDepth3[i]->Draw();
        }
        geometry->Draw();
        
        
    }

     void Animate(float tstart, float tend)
    {
        rotationAngle = 2.4f; //saját tengely körüli forgás
        rotationAxis = 1;

        
        dT += tend-tstart;
        if(dT > 0.1f)
        {
            dT = 0.0f;
            //Bown movement
            float correctionFactor = 0.05f;

            //parameter X
            float lower = -0.10f;
            float higher = 0.10f;
            if(isxPressed) { lower += correctionFactor;  higher += correctionFactor; }
            else if (isXPressed){  lower -= correctionFactor;   higher -= correctionFactor; }
            float randomX = lower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(higher-lower)));
            
            //parameter Y
            lower = -0.10f; //reinit
            higher = 0.10f; //reinit
            if(isyPressed) { lower += correctionFactor;  higher += correctionFactor; }
            else if(isYPressed){  lower -= correctionFactor;   higher -= correctionFactor; }
            float randomY = lower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(higher-lower)));
            
            //parameter Z
            lower = -0.10f;
            higher = 0.10f;
            if(iszPressed) { lower += correctionFactor;  higher += correctionFactor; }
            else if(isZPressed){  lower -= correctionFactor;   higher -= correctionFactor; }
            float randomZ = lower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(higher-lower)));

            vec3 speedVector =  vec3(randomX, randomY, randomZ);
            translation = translation + speedVector;

        }
        
        antibodyParent->reCreate(tessellationLevel, tessellationLevel, tend);
        createChildrenForParentTetrahedron(tend);
        createChildrenForDepth1Tetrahedrons(tend);
        //createChildrenForDepth2Tetrahedrons(tend);
    }
    
    
    void createChildrenForParentTetrahedron(float tend)
    {
        while(childrenDepth1.size() != 0){ childrenDepth1.pop_back();}
        
        
        
        for(int i = 2; i < antibodyParent->vtxData.size(); i++)
        {
            vec3 newTetrahedronVert1 = (antibodyParent->vtxData[i-2].position+antibodyParent->vtxData[i-1].position)/2.0f;
            vec3 newTetrahedronVert2 = (antibodyParent->vtxData[i-1].position+antibodyParent->vtxData[i].position)/2.0f;
            vec3 newTetrahedronVert3 = (antibodyParent->vtxData[i-2].position+antibodyParent->vtxData[i].position)/2.0f;
            Tetrahedron* child = new Tetrahedron(newTetrahedronVert1, newTetrahedronVert2,newTetrahedronVert3, 0.8f +  (childPath1Height * fabs(cos(tend))), antibodyParent->vtxData[i].normal);
            childrenDepth1.push_back(child);
        }
    }
    
    void createChildrenForDepth1Tetrahedrons(float tend)
    {
        while(childrenDepth2.size() != 0){ childrenDepth2.pop_back();}

        
        for(int j = 0; j < childrenDepth1.size(); j++)
        {
            for(int i = 2; i < childrenDepth1[j]->vtxData.size(); i++)
            {
                vec3 newTetrahedronVert1 = (childrenDepth1[j]->vtxData[i-2].position+childrenDepth1[j]->vtxData[i-1].position)/2.0f;
                vec3 newTetrahedronVert2 = (childrenDepth1[j]->vtxData[i-1].position+childrenDepth1[j]->vtxData[i].position)/2.0f;
                vec3 newTetrahedronVert3 = (childrenDepth1[j]->vtxData[i-2].position+childrenDepth1[j]->vtxData[i].position)/2.0f;
                childrenDepth2.push_back(new Tetrahedron(newTetrahedronVert1, newTetrahedronVert2,newTetrahedronVert3, 0.2f + (childPath2Height * fabs(cos(tend))), childrenDepth1[j]->vtxData[i].normal));
            }
            
        }
        
    }
    
    void createChildrenForDepth2Tetrahedrons(float tend)
    {
        while(childrenDepth3.size() != 0){ childrenDepth3.pop_back();}
        
        for(int j = 0; j < childrenDepth2.size(); j++)
        {
            for(int i = 2; i < childrenDepth2[j]->vtxData.size(); i++)
            {
                vec3 newTetrahedronVert1 = (childrenDepth2[j]->vtxData[i-2].position+childrenDepth2[j]->vtxData[i-1].position)/2.0f;
                vec3 newTetrahedronVert2 = (childrenDepth2[j]->vtxData[i-1].position+childrenDepth2[j]->vtxData[i].position)/2.0f;
                vec3 newTetrahedronVert3 = (childrenDepth2[j]->vtxData[i-2].position+childrenDepth2[j]->vtxData[i].position)/2.0f;
                childrenDepth3.push_back(new Tetrahedron(newTetrahedronVert1, newTetrahedronVert2,newTetrahedronVert3, 0.1f + (childPath2Height * fabs(cos(tend))), childrenDepth2[j]->vtxData[i].normal));
            }
            
        }
        
    }
    
};


//---------------------------
class Scene {
//---------------------------
    std::vector<Object *> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
public:
    void Build() {
        // Shaders
        Shader * phongShader = new PhongShader();
        Shader * gouraudShader = new GouraudShader();
        Shader * nprShader = new NPRShader();

        // Materials
        Material * material0 = new Material;
        material0->kd = vec3(0.6f, 0.4f, 0.2f);
        material0->ks = vec3(4, 4, 4);
        material0->ka = vec3(0.1f, 0.1f, 0.1f);
        material0->shininess = 100;

        Material * material1 = new Material;
        material1->kd = vec3(0.8f, 0.6f, 0.4f);
        material1->ks = vec3(0.3f, 0.3f, 0.3f);
        material1->ka = vec3(0.2f, 0.2f, 0.2f);
        material1->shininess = 30;

        // Textures
        Texture * texture4x8 = new CheckerBoardTexture(4, 8); //parameterezheto sakktabla textura
        Texture * texture15x20 = new CheckerBoardTexture(15, 20); //mas parameterekkel
        Texture * stripyTexture = new StripyTexture(150, 150);
        Texture * rainbowTexture = new RainbowTexture(200,200);


        // Geometries
        VirusParent * virusParent = new VirusParent();
        Geometry * virusParentGeometry = virusParent;
        
        Geometry * tractricoid = new Tractricoid();
        
        //tetrahedron geometry
        vec3 p3 = vec3(2.0f, 0.0f, 0.0f);
        vec3 p1 = vec3(0.0f, 0.0f, 0.0f);
        vec3 p2 = vec3(1.0f, 0.0f, -1.3f);
        Tetrahedron * tetrahedron = new Tetrahedron(p1,p2,p3,1.0f,vec3(0.0f,1.0f,0.0f));
        Geometry * tetrahedronGeometry = tetrahedron;
        
        // Create objects by setting up their vertex data on the GPU
        //Antibody object
        Object * antibodyObject = new AntibodyObject(gouraudShader, material1, texture15x20, tetrahedronGeometry, tetrahedron);
        antibodyObject->translation = vec3(-3, 2.5, 0);
        //sphereObject1->rotationAxis = vec3(0, 1, 1);
        antibodyObject->scale = vec3(0.5f, 0.5f, 0.5f);
        objects.push_back(antibodyObject);
        
        //MARK: a virus es antitest object együtt nem működik, a virus felülírja az antitest objectet
        //Virus object
        Object * virusObject = new VirusObject(phongShader, material0, stripyTexture, virusParentGeometry, virusParent);
        virusObject->translation = vec3(3, 0, 0);
        //virusObject->rotationAxis = vec3(1, 1, -1);
        virusObject->scale = vec3(0.7f, 0.7f, 0.7f);
        objects.push_back(virusObject);
        
    
        
        // Camera
        camera.wEye = vec3(0, 0, 6);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(0, 1, 0);

        // Lights
        lights.resize(3);
        lights[0].wLightPos = vec4(5, 5, 4, 0);    // ideal point -> directional light source
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(3, 0, 0);

        lights[1].wLightPos = vec4(5, 10, 20, 0);    // ideal point -> directional light source
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(0, 3, 0);

        lights[2].wLightPos = vec4(-5, 5, 5, 0);    // ideal point -> directional light source
        lights[2].La = vec3(0.1f, 0.1f, 0.1f);
        lights[2].Le = vec3(0, 0, 3);
    }

    void Render() {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object * obj : objects) obj->Draw(state);
    }

    void Animate(float tstart, float tend) {
        camera.Animate(tend);
        for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
        for (Object * obj : objects) obj->Animate(tstart, tend);
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY)
{
    switch (key)
    {
        case 'x': isxPressed = true; break;
        case 'X': isXPressed = true; break;
        case 'y': isyPressed = true; break;
        case 'Y': isYPressed = true; break;
        case 'z': iszPressed = true; break;
        case 'Z': isZPressed = true; break;

    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY)
{
    switch (key)
    {
        case 'x': isxPressed = false; break;
        case 'X': isXPressed = false; break;
        case 'y': isyPressed = false; break;
        case 'Y': isYPressed = false; break;
        case 'z': iszPressed = false; break;
        case 'Z': isZPressed = false; break;

    }

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is infinitesimal
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
