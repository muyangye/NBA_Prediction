import csv


# Get rid of spaces and newline characters
def strip(s):
    return s.strip()


def main():
    results = []
    with open("2020_2021_results.txt") as f:
        lines = list(map(strip, f.readlines()))
        for line in lines:
            visit_team, visit_points, home_team, home_points = line.split(",")
            home_points = int(home_points)
            visit_points = int(visit_points)
            win_team = home_team if home_points > visit_points else visit_team
            lose_team = home_team if home_points < visit_points else visit_team
            win_loc = "H" if home_points > visit_points else "V"
            results.append((win_team, lose_team, win_loc))
    header = ["WinTeam", "LoseTeam", "WinLoc"]
    with open("2020_2021_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


if __name__ == "__main__":
    main()
