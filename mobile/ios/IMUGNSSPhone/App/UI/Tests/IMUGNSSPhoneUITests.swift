import XCTest

final class IMUGNSSPhoneUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false

        app = XCUIApplication()
        app.launchArguments.append("-ui-testing")
        app.launch()
    }

    override func tearDown() {
        app = nil
        super.tearDown()
    }

    func testDriveMapControlsToggleVisibleState() {
        openTab("Drive")

        let showAccuracy = app.buttons["Show accuracy overlay"]
        XCTAssertTrue(showAccuracy.waitForExistence(timeout: 5))
        showAccuracy.tap()
        XCTAssertTrue(app.buttons["Hide accuracy overlay"].waitForExistence(timeout: 2))

        let stopFollowing = app.buttons["Stop following marker"]
        XCTAssertTrue(stopFollowing.waitForExistence(timeout: 2))
        stopFollowing.tap()
        XCTAssertTrue(app.buttons["Follow marker"].waitForExistence(timeout: 2))

        app.buttons["Resize telemetry drawer"].tap()
        XCTAssertTrue(app.buttons["Metrics"].waitForExistence(timeout: 2))
        XCTAssertTrue(app.buttons["Streams"].waitForExistence(timeout: 2))
    }

    func testDriveStreamAndRecordingControlsUpdateLabels() {
        openTab("Drive")

        let startStream = app.buttons["Start data stream"]
        XCTAssertTrue(startStream.waitForExistence(timeout: 5))
        startStream.tap()

        let stopStream = app.buttons["Stop data stream"]
        XCTAssertTrue(stopStream.waitForExistence(timeout: 2))

        let startLogging = app.buttons["Start raw data logging"]
        XCTAssertTrue(startLogging.waitForExistence(timeout: 2))
        XCTAssertTrue(startLogging.isEnabled)
        startLogging.tap()

        let stopLogging = app.buttons["Stop raw data logging"]
        XCTAssertTrue(stopLogging.waitForExistence(timeout: 2))
        stopLogging.tap()
        XCTAssertTrue(app.buttons["Start raw data logging"].waitForExistence(timeout: 2))

        stopStream.tap()
        XCTAssertTrue(app.buttons["Start data stream"].waitForExistence(timeout: 2))
    }

    func testSettingsRememberMountToggleUpdatesDisplayedMode() {
        openTab("Settings")

        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 5))

        let rememberMount = app.switches["rememberMountToggle"]
        scrollTo(rememberMount)
        XCTAssertTrue(rememberMount.waitForExistence(timeout: 2))
        XCTAssertTrue(waitForLabel(identifier: "mountModeRow", containing: "Auto Align", timeout: 2))
        XCTAssertTrue(waitForLabel(identifier: "savedMountRow", containing: "None", timeout: 2))

        tapSwitch(rememberMount)
        XCTAssertTrue(waitForLabel(identifier: "mountModeRow", containing: "Auto Align, Saving Mount", timeout: 2))

        tapSwitch(rememberMount)
        XCTAssertTrue(waitForLabel(identifier: "mountModeRow", containing: "Auto Align", timeout: 2))
    }

    func testDeveloperDiagnosticsToggleControlsDiagnosticsTab() {
        openTab("Settings")

        XCTAssertFalse(app.tabBars.buttons["Diagnostics"].exists)

        let diagnosticsToggle = app.switches["developerDiagnosticsToggle"]
        scrollTo(diagnosticsToggle)
        XCTAssertTrue(diagnosticsToggle.waitForExistence(timeout: 2))
        tapSwitch(diagnosticsToggle)

        let diagnosticsTab = app.tabBars.buttons["Diagnostics"]
        XCTAssertTrue(diagnosticsTab.waitForExistence(timeout: 2))
        diagnosticsTab.tap()
        XCTAssertTrue(app.navigationBars["Diagnostics"].waitForExistence(timeout: 2))

        openTab("Settings")
        scrollTo(diagnosticsToggle)
        tapSwitch(diagnosticsToggle)
        XCTAssertTrue(diagnosticsTab.waitForNonExistence(timeout: 2))
    }

    private func openTab(_ title: String) {
        let tab = app.tabBars.buttons[title]
        XCTAssertTrue(tab.waitForExistence(timeout: 5), "Missing tab: \(title)")
        tab.tap()
    }

    private func scrollTo(_ element: XCUIElement, maxSwipes: Int = 6) {
        for _ in 0 ..< maxSwipes where !element.exists || !element.isHittable {
            app.swipeUp()
        }
    }

    private func tapSwitch(_ element: XCUIElement) {
        element.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
    }

    private func waitForLabel(identifier: String, containing text: String, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            let element = app.descendants(matching: .any)[identifier]
            if element.exists, element.label.contains(text) {
                return true
            }
            Thread.sleep(forTimeInterval: 0.1)
        } while Date() < deadline
        return false
    }
}

private extension XCUIElement {
    func waitForNonExistence(timeout: TimeInterval) -> Bool {
        let predicate = NSPredicate(format: "exists == false")
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: self)
        return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
    }
}
