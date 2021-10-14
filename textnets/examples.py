# -*- coding: utf-8 -*-

"""Example data for feature demonstrations."""

from pandas import Series

#: Example dataset with newspaper headlines about the Apollo 11 landing.
moon_landing = Series(
    [
        "3:56 am: Man Steps On to the Moon",
        "Men Walk on Moon -- Astronauts Land on Plain, Collect Rocks, Plant Flag",
        "Man Walks on Moon",
        'Armstrong and Aldrich "Take One Small Step for Man" on the Moon',
        "The Eagle Has Landed -- Two Men Walk on the Moon",
        "Giant Leap for Mankind -- Armstrong Takes 1st Step on Moon",
        "Walk on Moon -- That's One Small Step for Man, One Giant Leap for Mankind",
    ],
    index=[
        "The Guardian",
        "New York Times",
        "Boston Globe",
        "Houston Chronicle",
        "Washington Post",
        "Chicago Tribune",
        "Los Angeles Times",
    ],
    name="headlines",
    dtype="object",
)

#: Example dataset with statements by five major German parties on digitization
#: of higher education during the 2021 Bundestag election.
digitalisierung = Series(
    {
        "cdu/csu": """Mit der Bund-Länder-Vereinbarung „Innovation in der
            Hochschullehre“ unterstützen CDU und CSU die qualitative
            Verbesserung der Hochschullehre. Dazu soll die Entwicklung
            innovativer Studien- und Lehrformate intensiv gefördert und die
            Projektergebnisse über eine Plattform gebündelt und allen
            Hochschulen für die Anwendung zugänglich gemacht werden. Gerade
            auch die gesammelten Erfahrungen während der Corona-Pandemie sollen
            dazu genutzt werden, Schwachstellen zu identifizieren und
            Lösungsansätze zu erarbeiten. Darüber hinaus dienen die Mittel des
            Zukunftsvertrags „Studium und Lehre stärken“ unter anderem auch der
            Erweiterung digitaler Angebote in der Lehre ebenso wie dem Ausbau
            der digitalen Infrastruktur an den Hochschulen. Außerdem
            unterstützt das unionsgeführte Bundesministerium für Bildung und
            Forschung das Hochschulforum Digitalisierung bis 2025 mit 15
            Millionen Euro.""",
        "spd": """Wir wollen eine Digitalisierungspauschale einführen, die
            Hochschulen mit einem verlässlichen Mittelumfang bei der
            Digitalisierung ihrer Infrastruktur entlang ihrer individuellen
            Bedarfe unterstützt. Wir wollen die besondere Leistung der
            Hochschulen in der Pandemie, bei Lehre, Studium und Forschung so
            sichern, dass die guten Beispiele und gemachten Erfahrungen für die
            Zukunft zum Tragen kommen.""",
        "fdp": """Wir Freie Demokraten fordern eine Qualitätsoffensive für die
            Hochschullehre mit der wir durch eine bundesweite Beratung
            Hochschulen und Lehrende bei didaktischen, technischen,
            datenschutz- und urheberrechtlichen Fragen zu digitaler Lehre
            unterstützen. Das starre Kapazitätsrecht, das die Zahl der
            bereitgestellten Studienplätze regelt, wollen wir grundlegend
            reformieren, um Hochschulen mehr Investitionen in digitale
            Lehrangebote, bessere Betreuungsquoten sowie berufs- und
            lebensbegleitende Studienmodule zu ermöglichen. Um Hochschulen im
            digitalen Zeitalter zu schützen, braucht es eine Nationale
            Strategie für Cybersicherheit in der Wissenschaft. Die Fraktion der
            Freien Demokraten im Deutschen Bundestag hat die Corona-Krise auch
            als Chance gesehen, um ohnehin längst überfällige strukturelle
            Innovationen im Hochschulbetrieb voranzutreiben (vgl.
            „Corona-Sofortprogramm für eine digitale und flexible
            Hochschullehre“ BT-Drs.-19/19121). Wir setzen uns zudem für die
            Schaffung einer Bundeszentrale für digitale Bildung ein. Eine
            wesentliche Aufgabe besteht darin, die digitale Transformation des
            Bildungswesens zu stärken, mit besonderem Augenmerk auf digitaler
            Didaktik und der Aus- sowie Fortbildung von Lehrenden aller
            Bildungsinstitutionen zur Implementierung digitaler
            Lernstrategien.""",
        "linke": """DIE LINKE will für einen schnelleren Aus- und Aufbau
            digitaler Infrastrukturen an den Hochschulen von Bund und Ländern
            zusätzliche finanzielle Mittel durch einen Hochschuldigitalpakt zur
            Verfügung stellen. Statt Leuchtturmprojekten braucht es eine
            bundesweite Digitalisierungsoffensive an den Hochschulen. Doch
            Technik allein macht noch keine gute Onlinelehre. Lehrenden muss
            der Zugang zu Fort- und Weiterbildung für digitale Lehr- und
            Lernangebote erleichtert werden.""",
        "gruene": """Wir GRÜNE wollen an Hochschulen eine nachhaltige,
            klimagerechte und barrierefreie Modernisierung ermöglichen, die
            auch digitale Infrastruktur und IT-Sicherheit mit einschließt. Wir
            werden sie dabei unterstützen, neue Lösungen für den Klimaschutz zu
            entwickeln und als Reallabore Ideen für Klimaneutralität praktisch
            vor Ort zu erproben. Wir werden über eine Digitalisierungspauschale
            die IT-Infrastruktur an Hochschulen stärken und die
            IT-Barrierefreiheit einfordern, Aus- und Weiterbildung der
            Lehrenden ausbauen und digitale Beratungs- und Betreuungsangebote
            für Studierende ausweiten. Der Zugang zu Forschungs- und
            Bildungsdaten soll erleichtert und FAIR Data das Grundprinzip
            werden. Wir wollen zudem Open Access bei Publikationen zum Standard
            erklären und als wissenschaftliche Leitidee stärker fördern und
            zusammen mit der Wissenschaft vorantreiben. Die dadurch anstehende
            Reform der Finanzierung wissenschaftlicher Publikationen darf nicht
            zu Lasten der Forscher*innen oder ihrer Einrichtungen gehen.""",
    },
    name="statements",
    dtype="object",
)
