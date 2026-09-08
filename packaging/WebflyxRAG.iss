#define MyAppName "Webflyx RAG Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Alex Unnippillil"
#define MyAppExeName "WebflyxRAG.exe"

[Setup]
AppId={{76DF72C7-0E42-450B-A185-DAF450632F21}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Webflyx RAG Studio
DefaultGroupName={#MyAppName}
OutputDir=..\dist-installer
OutputBaseFilename=Webflyx-RAG-Studio-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\webflyx.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\WebflyxRAG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Webflyx RAG Studio"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\Webflyx RAG Studio"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Webflyx RAG Studio"; Flags: nowait postinstall skipifsilent
