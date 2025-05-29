import re
import string
import tldextract
from urllib.parse import urlparse
import pycountry_convert as pc

class FeatureExtractor:

    @staticmethod
    def get_url_length(url):
        prefixes = ['http://', 'https://']
        for prefix in prefixes:
            if url.startswith(prefix):
                url = url[len(prefix):]
        url = url.replace('www.', '')
        return len(url)

    @staticmethod
    def extract_root_domain(url):
        extracted = tldextract.extract(url)
        return extracted.domain

    @staticmethod
    def root_domain_length(root_domain):
        return len(root_domain) if root_domain else 0

    @staticmethod
    def hostname_length(url):
        return len(urlparse(url).netloc)

    @staticmethod
    def get_subdomain(url):
        return tldextract.extract(url).subdomain

    @staticmethod
    def subdomain_length(url):
        subdomain = FeatureExtractor.get_subdomain(url)
        return len(subdomain) if subdomain else 0

    @staticmethod
    def pri_domain_length(url):
        if url:
            return len(urlparse(url).netloc.split('.')[0])
        return 0

    @staticmethod
    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go21\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|ur14\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|o\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          r'tr\.im|link\.zip\.net',
                          url)
        return 1 if match else 0

    @staticmethod
    def count_special_chars(url):
        special_chars = set(string.punctuation)
        return sum(char in special_chars for char in url)

    @staticmethod
    def count_digits(url):
        return sum(char.isdigit() for char in url)

    @staticmethod
    def count_letters(url):
        return sum(char.isalpha() for char in url)

    @staticmethod
    def secure_http(url):
        return int(urlparse(url).scheme == 'https')

    @staticmethod
    def having_ip_address(url):
        ip_pattern = re.compile(
            r'((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}'
            r'(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)'                       # IPv4
            r'|((0x[0-9a-fA-F]{1,2})\.){3}(0x[0-9a-fA-F]{1,2})'         # IPv4 in hex
            r'|([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}'                  # IPv6
        )
        return 1 if ip_pattern.search(url) else 0

    @staticmethod
    def get_url_region(primary_domain):
        ccTLD_to_region = {
            ".ac": "Ascension Island",".ad": "Andorra",".ae": "United Arab Emirates",".af": "Afghanistan",".ag": "Antigua and Barbuda",".ai": "Anguilla",
            ".al": "Albania",".am": "Armenia",".an": "Netherlands Antilles",".ao": "Angola",".aq": "Antarctica",".ar": "Argentina",".as": "American Samoa",
            ".at": "Austria",".au": "Australia",".aw": "Aruba",".ax": "Åland Islands",".az": "Azerbaijan",".ba": "Bosnia and Herzegovina",".bb": "Barbados",
            ".bd": "Bangladesh",".be": "Belgium",".bf": "Burkina Faso",".bg": "Bulgaria",".bh": "Bahrain",".bi": "Burundi",".bj": "Benin",".bm": "Bermuda",
            ".bn": "Brunei Darussalam",".bo": "Bolivia",".br": "Brazil",".bs": "Bahamas",".bt": "Bhutan",".bv": "Bouvet Island",".bw": "Botswana",".by": "Belarus",
            ".bz": "Belize",".ca": "Canada",".cc": "Cocos Islands",".cd": "Democratic Republic of the Congo",".cf": "Central African Republic",".cg": "Republic of the Congo",
            ".ch": "Switzerland",".ci": "Côte d'Ivoire",".ck": "Cook Islands",".cl": "Chile",".cm": "Cameroon",".cn": "China",".co": "Colombia",".cr": "Costa Rica",
            ".cu": "Cuba",".cv": "Cape Verde",".cw": "Curaçao",".cx": "Christmas Island",".cy": "Cyprus",".cz": "Czech Republic",".de": "Germany",".dj": "Djibouti",
            ".dk": "Denmark",".dm": "Dominica",".do": "Dominican Republic",".dz": "Algeria",".ec": "Ecuador",".ee": "Estonia",".eg": "Egypt",".er": "Eritrea",
            ".es": "Spain",".et": "Ethiopia",".eu": "European Union",".fi": "Finland",".fj": "Fiji",".fk": "Falkland Islands",".fm": "Federated States of Micronesia",".fo": "Faroe Islands",
            ".fr": "France",".ga": "Gabon",".gb": "United Kingdom",".gd": "Grenada",".ge": "Georgia",".gf": "French Guiana",".gg": "Guernsey",".gh": "Ghana",
            ".gi": "Gibraltar",".gl": "Greenland",".gm": "Gambia",".gn": "Guinea",".gp": "Guadeloupe",".gq": "Equatorial Guinea",".gr": "Greece",".gs": "South Georgia and the South Sandwich Islands",
            ".gt": "Guatemala",".gu": "Guam",".gw": "Guinea-Bissau",".gy": "Guyana",".hk": "Hong Kong",".hm": "Heard Island and McDonald Islands",".hn": "Honduras",".hr": "Croatia",".ht": "Haiti",
            ".hu": "Hungary",".id": "Indonesia",".ie": "Ireland",".il": "Israel",".im": "Isle of Man",".in": "India",".io": "British Indian Ocean Territory",".iq": "Iraq",".ir": "Iran",".is": "Iceland",
            ".it": "Italy",".je": "Jersey",".jm": "Jamaica",".jo": "Jordan",".jp": "Japan",".ke": "Kenya",".kg": "Kyrgyzstan",".kh": "Cambodia",".ki": "Kiribati",".km": "Comoros",".kn": "Saint Kitts and Nevis",
            ".kp": "Democratic People's Republic of Korea (North Korea)",".kr": "Republic of Korea (South Korea)",".kw": "Kuwait",".ky": "Cayman Islands",".kz": "Kazakhstan",".la": "Laos",".lb": "Lebanon",
            ".lc": "Saint Lucia",".li": "Liechtenstein",".lk": "Sri Lanka",".lr": "Liberia",".ls": "Lesotho",".lt": "Lithuania",".lu": "Luxembourg",".lv": "Latvia",".ly": "Libya",".ma": "Morocco",".mc": "Monaco",
            ".md": "Moldova",".me": "Montenegro",".mf": "Saint Martin (French part)",".mg": "Madagascar",".mh": "Marshall Islands",".mk": "North Macedonia",".ml": "Mali",".mm": "Myanmar",".mn": "Mongolia",
            ".mo": "Macao",".mp": "Northern Mariana Islands",".mq": "Martinique",".mr": "Mauritania",".ms": "Montserrat",".mt": "Malta",".mu": "Mauritius",".mv": "Maldives",".mw": "Malawi",".mx": "Mexico",
            ".my": "Malaysia",".mz": "Mozambique",".na": "Namibia",".nc": "New Caledonia",".ne": "Niger",".nf": "Norfolk Island",".ng": "Nigeria",".ni": "Nicaragua",".nl": "Netherlands",".no": "Norway",".np": "Nepal",
            ".nr": "Nauru",".nu": "Niue",".nz": "New Zealand",".om": "Oman",".pa": "Panama",".pe": "Peru",".pf": "French Polynesia",".pg": "Papua New Guinea",".ph": "Philippines",".pk": "Pakistan",".pl": "Poland",
            ".pm": "Saint Pierre and Miquelon",".pn": "Pitcairn",".pr": "Puerto Rico",".ps": "Palestinian Territory",".pt": "Portugal",".pw": "Palau",".py": "Paraguay",".qa": "Qatar",".re": "Réunion",".ro": "Romania",
            ".rs": "Serbia",".ru": "Russia",".rw": "Rwanda",".sa": "Saudi Arabia",".sb": "Solomon Islands",".sc": "Seychelles",".sd": "Sudan",".se": "Sweden",".sg": "Singapore",".sh": "Saint Helena",".si": "Slovenia",
            ".sj": "Svalbard and Jan Mayen",".sk": "Slovakia",".sl": "Sierra Leone",".sm": "San Marino",".sn": "Senegal",".so": "Somalia",".sr": "Suriname",".ss": "South Sudan",".st": "São Tomé and Príncipe",
            ".sv": "El Salvador",".sx": "Sint Maarten (Dutch part)",".sy": "Syria",".sz": "Eswatini",".tc": "Turks and Caicos Islands",".td": "Chad",".tf": "French Southern Territories",".tg": "Togo",".th": "Thailand",
            ".tj": "Tajikistan",".tk": "Tokelau",".tl": "Timor-Leste",".tm": "Turkmenistan",".tn": "Tunisia",".to": "Tonga",".tr": "Turkey",".tt": "Trinidad and Tobago",".tv": "Tuvalu",".tw": "Taiwan",".tz": "Tanzania",
            ".ua": "Ukraine",".ug": "Uganda",".uk": "United Kingdom",".us": "United States",".uy": "Uruguay",".uz": "Uzbekistan",".va": "Vatican City",".vc": "Saint Vincent and the Grenadines",".ve": "Venezuela",
            ".vg": "British Virgin Islands",".vi": "U.S. Virgin Islands",".vn": "Vietnam",".vu": "Vanuatu",".wf": "Wallis and Futuna",".ws": "Samoa",".ye": "Yemen",".yt": "Mayotte",".za": "South Africa",".zm": "Zambia",
            ".zw": "Zimbabwe"
        }
        for ccTLD in ccTLD_to_region:
            if primary_domain.endswith(ccTLD):
                return ccTLD_to_region[ccTLD]
        return "Global"

    @staticmethod
    def get_continent(country_name):
        try:
            country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            return {
                'AF': 'Africa',
                'AS': 'Asia',
                'EU': 'Europe',
                'NA': 'North America',
                'SA': 'South America',
                'OC': 'Oceania',
                'AN': 'Antarctica'
            }[continent_code]
        except:
            return 'Unknown'

    @staticmethod
    def abnormal_url(url):
        hostname = str(urlparse(url).hostname)
        return 1 if re.search(hostname, url) else 0

    @staticmethod
    def no_of_dir(url):
        return urlparse(url).path.count('/')

    @staticmethod
    def custom_hash_encode(category):
        hash_value = 5381
        for char in category:
            hash_value = ((hash_value << 5) + hash_value) + ord(char)
        return hash_value % (10 ** 8)

    @staticmethod
    def extract(url):
        root_domain = FeatureExtractor.extract_root_domain(url)
        primary_domain = urlparse(url).netloc
        region = FeatureExtractor.get_url_region(primary_domain)
        continent = FeatureExtractor.get_continent(region)

        extraction = {
            'url_len': FeatureExtractor.get_url_length(url),
            'root_domain_length': FeatureExtractor.root_domain_length(root_domain),
            'hostname_length': FeatureExtractor.hostname_length(url),
            'subdomain_length': FeatureExtractor.subdomain_length(url),
            'short_url': FeatureExtractor.shortening_service(url),
            'num_special_chars': FeatureExtractor.count_special_chars(url),
            'num_digits': FeatureExtractor.count_digits(url),
            'num_letters': FeatureExtractor.count_letters(url),
            'secure_http': FeatureExtractor.secure_http(url),
            'use_of_ip': FeatureExtractor.having_ip_address(url),
            'abnormal_url': FeatureExtractor.abnormal_url(url),
            'count_dir': FeatureExtractor.no_of_dir(url),
            'hash_region': FeatureExtractor.custom_hash_encode(region),
            'hash_root_domain': FeatureExtractor.custom_hash_encode(root_domain if root_domain else ''),
            'hash_continent': FeatureExtractor.custom_hash_encode(continent)}

        return extraction
