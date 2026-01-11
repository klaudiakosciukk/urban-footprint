import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-cities", action="store_true")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2025])
    args = parser.parse_args()

    print("Pipeline placeholder")
    print("all_cities:", args.all_cities)
    print("city:", args.city)
    print("years:", args.years)

if __name__ == "__main__":
    main()
