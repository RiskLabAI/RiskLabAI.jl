using PkgTemplates

t = Template(; 
    user="hamid-arian",
    dir="~/Code/PkgLocation/",
    julia=v"1.8",
    plugins=[
        Git(; manifest=true, ssh=true),
        GitHubActions(; x86=true),
        Codecov(),
        Documenter{GitHubActions}(),
    ],
)

t("RiskLabAI")